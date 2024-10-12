
import math
import warnings
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaForCausalLM
from transformers.generation.utils import GenerateOutput

from constants import IGNORE_INDEX
from models.modeling import MonoFormerModel
from models.vision_encoder import build_vision_encoder


@dataclass
class MonoFormerCausalLMOutputWithPast(CausalLMOutputWithPast):
    x_out: Optional[torch.FloatTensor] = None
    x_indices: Optional[torch.LongTensor] = None


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear( hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32
            ) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([
                embedding, torch.zeros_like(embedding[:, :1])
            ], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


def modulate(x, shift, scale, mask):
    return x * (1 + scale.unsqueeze(1) * mask.unsqueeze(2)) + shift.unsqueeze(1) * mask.unsqueeze(2)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_adaln=True):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.use_adaln = use_adaln
        if self.use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True),
            )
        else:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )

    def forward(self, x, c, mask):
        if self.use_adaln:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale, mask)
        else:
            x = x + self.mlp(c).unsqueeze(1) * mask.unsqueeze(2)
        x = self.linear(x)
        return x


class MonoFormerForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MonoFormerModel(config)  # override the original llama model
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels * 2 if config.learn_sigma else self.in_channels

        self.x_embedder = nn.Linear(
            in_features=self.patch_size * self.patch_size * self.in_channels,
            out_features=config.hidden_size,
            bias=True,
        )

        if self.config.decoder_t_embed == 'add_with_embed_at_each_layer':
            self.t_projector = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
            )
        elif self.config.decoder_t_embed == 'add_before_decoder':
            self.t_projector = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
            )
        elif self.config.decoder_t_embed == 'add_at_each_layer':
            self.t_projector = nn.ModuleList([nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
            ) for _ in range(config.num_hidden_layers)])
        elif self.config.decoder_t_embed == 'adaln_before_decoder':
            self.adaLN_module = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True)
            )
        elif self.config.decoder_t_embed == 'add_before_image_tokens':
            self.t_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size, bias=True),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            )
        else:
            raise NotImplementedError(f"t_embed strategy {self.config.decoder_t_embed} not supported.")

        self.final_layer = FinalLayer(config.hidden_size, self.patch_size, self.out_channels, use_adaln=self.config.use_adaln_final_layer)
        
        self.vision_encoder = None
        if getattr(self.config, 'vision_encoder', None):
            self.initialize_vision_modules(self.config.vision_encoder, self.config.vision_encoder_path, self.config.vision_encoder_args)
        
    
    def initialize_weights(self):
        """
        Call this function to initialize the additional modules for DiT generation after loading pretrained weights.
        """
        for m in self.final_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.t_embedder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.constant_(self.x_embedder.weight, 0)
        nn.init.constant_(self.x_embedder.bias, 0)

        if hasattr(self, 't_projector'):
            for m in self.t_projector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        if hasattr(self, 'adaLN_module'):
            for m in self.adaLN_module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
    
    def initialize_vision_modules(self, vision_encoder: str, pretrained_path: str, model_args: Dict):
        self.config.vision_encoder = vision_encoder
        self.config.vision_encoder_path = pretrained_path
        self.config.vision_encoder_args = model_args

        self.vision_encoder = build_vision_encoder(self.config.vision_encoder, self.config.vision_encoder_path, **self.config.vision_encoder_args)

        self.im_embedder = nn.Sequential(
            nn.Linear(self.vision_encoder.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
    
    def parameter_count(self) -> int:
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.model.layers)

    def patchify_and_embed(self, x: List[torch.Tensor], embedder: nn.Module) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        if len(x) == 0:
            return [], []

        pH = pW = self.patch_size
        x_embed = []
        img_size = []

        for img in x:
            C, H, W = img.size()
            img_size.append((H, W))
            img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 0, 2, 4).flatten(2)
            img = self.x_embedder(img)
            img = img.flatten(0, 1)
            x_embed.append(img)
        
        x_embed = torch.stack(x_embed, dim=0)
        
        return x_embed, img_size
        
    def unpatchify(self, x: torch.Tensor, img_size: List[Tuple[int, int]], return_tensor=False) -> List[torch.Tensor]:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        pH = pW = self.patch_size
        if return_tensor:
            H, W = img_size[0]
            B = x.size(0)
            L = (H // pH) * (W // pW)
            x = x[:, :L].view(B, H // pH, W // pW, pH, pW, self.out_channels)
            x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
            return x
        else:
            imgs = []
            for i in range(x.size(0)):
                H, W = img_size[i]
                L = (H // pH) * (W // pW)
                imgs.append(x[i][:L].view(
                    H // pH, W // pW, pH, pW, self.out_channels
                ).permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2))
        return imgs

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images, x_t, flags, t, **kwargs
    ):
        """
        Args:
            images (list or tensor): list of image features of shape (C, H, W) or image tensor of shape (B, C, H, W)
        """
        # in the next token generation process of inference
        if input_ids.shape[1] == 1:
            model_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                'inputs_embeds': None,
                "labels": labels,
                'image_token_spans': None,
                'image_sizes': None,
                'c_embeds': None,
                't_embeds': None,
                'c_embeds_mask': None,
            }
            return model_inputs
        
        if images is None:
            images = []

        x_t_inputs = []
        img_inputs = []
        for bid in range(len(flags)):
            for i in range(len(flags[bid])):
                if flags[bid][i] == 0:
                    x_t_inputs.append(x_t[bid])
        for img_list in images:
            for img in img_list:
                img_inputs.append(img)

        x_t_embeds, x_t_sizes = self.patchify_and_embed(x_t_inputs, self.x_embedder)  # x_t_embeds: (N_x, max_seq_len, hidden_size)

        if getattr(self, 'im_embedder', None) is not None and len(img_inputs) > 0:
            image_features = self.vision_encoder(img_inputs)
            image_features = self.im_embedder(image_features)
        else:
            image_features = []
        
        # rearrange image embeddings according to the image order in conversation
        image_embeds = []
        image_sizes = []
        img_gen_idx = 0
        img_und_idx = 0
        for bid in range(len(flags)):
            image_size = None
            for i in range(len(flags[bid])):
                if flags[bid][i] == 0:
                    image_embeds.append(x_t_embeds[img_gen_idx])
                    image_size = x_t_sizes[img_gen_idx]
                    img_gen_idx += 1
                elif flags[bid][i] == 1:
                    image_embeds.append(image_features[img_und_idx])
                    img_und_idx += 1
            image_sizes.append(image_size)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # FIXME: remove paddings added in the collator function, this is trivial, fix later
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        # TODO: we need to differentiate between image for understanding and image for generation, considering add a flag to indicate in image_token_spans
        image_token_spans = []
        cur_image_idx = 0
        
        # compute the time embedding, only noise image tokens have valid time embedding, other tokens have time embeddding set to 0
        t_embeds = self.t_embedder(t)  # (batch_size, hidden_size)

        if self.config.decoder_t_embed == 'add_before_image_tokens':
            t_tokens = self.t_projector(t_embeds)  # (batch_size, hidden_size)

        # TODO: currently we expect only one image exists in each conversation, need to support multiple image in future
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == self.config.image_token_index).sum()
            if num_images == 0 or len(flags[batch_idx]) == 0:
                cur_input_embeds = self.model.embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                image_token_spans.append([])
                continue
            
            image_token_indices = [-1] + torch.where(cur_input_ids == self.config.image_token_index)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.model.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_image_token_spans = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    if self.config.decoder_t_embed == 'add_before_image_tokens':
                        cur_new_input_embeds.append(t_tokens[batch_idx:batch_idx+1])
                        cur_new_labels.append(torch.full((1,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_image_features = image_embeds[cur_image_idx]
                    cur_image_idx += 1
                    image_token_start_idx = torch.cat(cur_new_input_embeds).shape[0]
                    cur_new_input_embeds.append(cur_image_features)
                    image_token_end_idx = torch.cat(cur_new_input_embeds).shape[0]
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    # NOTE: in each sample at most one image are generated
                    if flags[batch_idx][i] == 0:
                        cur_image_token_spans = (image_token_start_idx, image_token_end_idx)

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            image_token_spans.append(cur_image_token_spans)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        #===================================
        #  generate condition embedding mask
        #===================================
        c_embeds_mask = torch.zeros(new_input_embeds.shape[:2]).to(t_embeds.device)  # (batch_size, seq_len)
        for i, span in enumerate(image_token_spans):
            if span:
                c_embeds_mask[i, span[0]:span[1]] = 1
        
        if self.config.decoder_t_embed in ['add_before_decoder', 'add_with_embed_at_each_layer', 'add_before_image_tokens']:
            c_embeds = self.t_projector(t_embeds)  # (batch_size, hidden_size)
        elif self.config.decoder_t_embed == 'add_at_each_layer':
            c_embeds = []
            for i in range(self.config.num_hidden_layers):
                c_embeds.append(self.t_projector[i](t_embeds))
            c_embeds = torch.stack(c_embeds)  # (num_decoder_layers, batch_size, hidden_size)
        elif self.config.decoder_t_embed == 'adaln_before_decoder':
            c_embeds = self.adaLN_module(t_embeds)
        else:
            raise NotImplementedError(f'time embedding strategy: {self.config.decoder_t_embed} is not implemented.')

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        model_inputs = {
            'input_ids': None,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': new_input_embeds,
            'labels': new_labels,
            'image_token_spans': image_token_spans,
            'image_sizes': image_sizes,
            'c_embeds': c_embeds,
            't_embeds': t_embeds,
            'c_embeds_mask': c_embeds_mask,
        }

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        images: Optional[List[List[torch.FloatTensor]]] = None,
        x_t: Optional[List[torch.FloatTensor]] = None,
        flags: Optional[List[torch.LongTensor]] = None,
        t: Optional[torch.FloatTensor] = None,
        # args for generation
        image_token_spans: Optional[List[Tuple[int, int]]] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        c_embeds: Optional[torch.FloatTensor] = None,
        t_embeds: Optional[torch.FloatTensor] = None,
        c_embeds_mask: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            images (Optional[List[List[torch.FloatTensor]]]): A list of size `batch_size`, where `images[i]` contains image(s) for image understanding corresponding to the sample at index `i`.
                If a sample does not contain images for understanding, `images[i]` will be an empty list.
            x_t (Optional[torch.FloatTensor]): A tensor of size `batch_size`, representing the noised images at timestep `t` used for diffusion generation. Each sample may have at most one image for generation.
                If a sample has no image for generation, `x_t[i]` is an empty tensor of shape `(0)`.
            flags (Optional[List[torch.LongTensor]]): A list of size `batch_size`, where `flags[i]` is a list of flags representing image types for the sample at index `i`.
                A value of `0` indicates an image for generation, while `1` indicates an image for understanding. Each image in a sample's conversation corresponds to a flag in `flags[i]`.
            t (Optional[torch.FloatTensor]): A tensor of size `batch_size`, where each value represents the timestep for the noised image `x_t` used in diffusion generation for the corresponding sample.
        """
        if inputs_embeds is None:
            model_inputs = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, x_t, flags, t)
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, image_token_spans, image_sizes, c_embeds, t_embeds, c_embeds_mask = (
                model_inputs['input_ids'], 
                model_inputs['position_ids'], 
                model_inputs['attention_mask'], 
                model_inputs['past_key_values'], 
                model_inputs['inputs_embeds'], 
                model_inputs['labels'],
                model_inputs['image_token_spans'],
                model_inputs['image_sizes'],
                model_inputs['c_embeds'],
                model_inputs['t_embeds'],
                model_inputs['c_embeds_mask'],
            )
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            c_embeds=c_embeds,
            c_embeds_mask=c_embeds_mask,
        )

        hidden_states = outputs['last_hidden_state']

        #=================================================
        # Compute outputs for diffusion generation losses
        #=================================================
        x_out = []
        x_indices = []
        if c_embeds_mask is not None:
            # t_embeds: (batch_size, hidden_size), c_embeds_mask: (batch_size, seq_len)
            x = self.final_layer(hidden_states, t_embeds, c_embeds_mask)
            for i in range(len(hidden_states)):
                span = image_token_spans[i]
                if span:
                    if span[1] > x.shape[1]:
                        warnings.warn(f'Number of image tokens: {span[1]} exceed maximum tokenizer length: {x.shape[1]}.')
                        continue
                    image_tokens = x[i, span[0]:span[1], :]
                    x_out.append(image_tokens)
                    x_indices.append(i)
        
        # only compute diffusion loss for samples in x_indices
        x_indices = torch.tensor(x_indices)
        if x_out:
            x_out = torch.stack(x_out)
            x_out = self.unpatchify(x_out, [image_sizes[i] for i in x_indices], return_tensor=False)

        #===================================
        # Compute the language modeling loss
        #===================================
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # compute loss for samples not used for diffusion generation
            lm_indices = torch.tensor([i for i in range(len(hidden_states)) if i not in x_indices], dtype=torch.long)
            if len(lm_indices) == 0:
                loss = torch.tensor(0., device=logits.device)
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[lm_indices, :-1, :].contiguous()
                shift_labels = labels[lm_indices, 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        
        return MonoFormerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            x_indices=x_indices,
            x_out=x_out,
        )

    def forward_with_cfg(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        images: Optional[List[torch.FloatTensor]] = None,
        x_t: Optional[torch.FloatTensor] = None,
        flags: Optional[List[List[torch.LongTensor]]] = None,
        t: Optional[torch.FloatTensor] = None,
        cfg_scale: float = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of DiT, but also batches the unconditional forward pass
        for classifier-free guidance.
        """
        half = x_t[: len(x_t) // 2]  # (batch_size, 256, dim)
        combined = torch.cat([half, half], dim=0)
        outputs = self.forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, images, combined, flags, t)
        x_out = outputs['x_out']
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        x_out = torch.stack(x_out)
        eps, rest = x_out[:, :3], x_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        outputs['x_out'] = torch.cat([eps, rest], dim=1)

        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        t = kwargs.pop("t", None)
        images = kwargs.pop("images", None)
        x_t = kwargs.pop("x_t", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        flags = kwargs.pop("flags", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        is_multimodal = False
        for i in range(len(flags)):
            if len(flags[i]) > 0:
                is_multimodal = True
                break
        
        if is_multimodal:
            model_inputs = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                x_t,
                flags,
                t,
            )
            _, position_ids, attention_mask, _, inputs_embeds, _, image_token_spans, image_sizes, c_embeds, t_embeds, c_embeds_mask = (
                model_inputs['input_ids'], 
                model_inputs['position_ids'], 
                model_inputs['attention_mask'], 
                model_inputs['past_key_values'], 
                model_inputs['inputs_embeds'], 
                model_inputs['labels'],
                model_inputs['image_token_spans'],
                model_inputs['image_sizes'],
                model_inputs['c_embeds'],
                model_inputs['t_embeds'],
                model_inputs['c_embeds_mask'],
            )
            kwargs.update({
                'image_token_spans': image_token_spans,
                'image_sizes': image_sizes,
                'c_embeds': c_embeds,
                't_embeds': t_embeds,
                'c_embeds_mask': c_embeds_mask
            })
        else:
            inputs_embeds = self.model.embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        attention_mask = kwargs.pop("attention_mask", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )
        inputs.update(kwargs)
        return inputs

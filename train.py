# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch FSDP.
"""
import shutil
import sys
import argparse
from collections import OrderedDict
import contextlib
from datetime import datetime
import functools
import json
import logging
import math
import os
import copy
import random
import socket
from time import time
from typing import Dict, List
import warnings

import numpy as np
from PIL import Image
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as torchdata
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoTokenizer, AutoConfig

from data import MixIterDataset, ImageNetDataset, JourneyDBDataset, UltraChatDataset, ToIterableDataset
from diffusion import create_diffusion
from imgproc import center_crop_arr

from config import read_config_from_file
from models import MonoFormerForCausalLM

from constants import DEFAULT_PAD_TOKEN, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN
import transformers
import base64
import io


logger = logging.getLogger(__name__)


class UltraChatItemProcessor:
    def __init__(self):
        pass

    def __call__(self, data_item: str):
        if isinstance(data_item, str):
            try:
                data_item = json.loads(data_item)
            except Exception as e:
                warnings.warn(f'failed to read data from ultrachat.')
                conversation  = {
                    'input': '',
                    'output': '',
                }
                return torch.empty(0), conversation, None
        
            # FIXME: currently, do not support multi-turn generation
            conversation = {}
            for i, conv in enumerate(data_item['data']):
                if i % 2 == 0:
                    conversation['input'] = conv
                else:
                    conversation['output'] = conv
        
        return torch.empty(0), conversation, None


class ImageNetItemProcessor:
    def __init__(self, transform, text_dropout_prob=0.1, annotation_path='./data/imagenet_labels.json'):
        self.image_transform = transform
        self.text_dropout_prob = text_dropout_prob
        self.annotation = json.load(open(annotation_path, 'r'))
    
    def __call__(self, data_item: Dict):
        image = Image.open(data_item['img_path']).convert("RGB")
        image = self.image_transform(image)

        label_id = data_item['label']
        anno = self.annotation[str(label_id)]
        label = anno['label']
        
        # NOTE: disable second labels
        # if len(anno['second_labels']) != 0 and random.random() < 0.5:
        #     label = random.choice(anno['second_labels'])
        
        # apply prompt template
        inputs = [
            f'Please generate an image of {label}.',
            f'{label}'
            f'an image of {label}.'
        ]

        conversation = {
            'input': random.choice(inputs),
            'output': DEFAULT_IMAGE_TOKEN + '\n',
        }

        # randomly convert some samples to unconditional generation
        if random.uniform(0., 1.) < self.text_dropout_prob and DEFAULT_IMAGE_TOKEN in conversation['output']:
            # NOTE: this is a hacky to replace all tokens in the input with padding token, so that these tokens are masked by attention mask in the collator.
            # but the number of tokens computed here is different from the number of tokens tokenized by tokenizer
            num_input_tokens = len(conversation['input'].split()) + conversation['input'].count(' ')
            conversation = {
                'input': DEFAULT_PAD_TOKEN * num_input_tokens,
                'output': DEFAULT_IMAGE_TOKEN + '\n',
            }
        
        noise_image_index = [0]

        # BUG: tokenizer can not properly tokenize '<|im_start|><image><|im_end|>', here we substitue <image> with <PAD> to make it tokenize correctly
        # We substitue <image> back after tokenizing in collate function
        conversation['input'] = conversation['input'].replace(DEFAULT_IMAGE_TOKEN, f'{DEFAULT_IMAGE_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_IMAGE_END_TOKEN}')
        conversation['output'] = conversation['output'].replace(DEFAULT_IMAGE_TOKEN, f'{DEFAULT_IMAGE_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_IMAGE_END_TOKEN}')

        return image, conversation, noise_image_index


class JourneyDBItemProcessor:
    def __init__(self, transform=None, text_dropout_prob=0.1, prompt_type='prompt', load_vae_feature=False):
        self.image_transform = transform
        self.text_dropout_prob = text_dropout_prob
        self.prompt_type = prompt_type
        self.load_vae_feature = load_vae_feature
        assert prompt_type in ['prompt', 'caption'], 'prompt_type must be either prompt or caption.'
    
    def __call__(self, data_item: Dict):
        image = data_item['image']
        if not self.load_vae_feature:
            image = self.image_transform(image)
        
        conversation = {
            'input': data_item[self.prompt_type],
            'output': DEFAULT_IMAGE_TOKEN + '\n',
        }

        conversation['input'] = f"Please generate an image of {conversation['input']}"
        
        # randomly convert some samples to unconditional generation
        if random.uniform(0., 1.) < self.text_dropout_prob and DEFAULT_IMAGE_TOKEN in conversation['output']:
            # NOTE: this is a hacky to replace all tokens in the input with padding token, so that these tokens are masked by attention mask in the collator.
            # but the number of tokens computed here is different from the number of tokens tokenized by tokenizer
            num_input_tokens = len(conversation['input'].split()) + conversation['input'].count(' ')
            conversation = {
                'input': DEFAULT_PAD_TOKEN * num_input_tokens,
                'output': DEFAULT_IMAGE_TOKEN + '\n',
            }
        
        # differentiate between image-generation and text-generation tasks
        noise_image_index = [0]

        # BUG: tokenizer can not properly tokenize '<|im_start|><image><|im_end|>', here we substitue <image> with <PAD> to make it tokenize correctly
        # We substitue <image> back after tokenizing in collate function
        conversation['input'] = conversation['input'].replace(DEFAULT_IMAGE_TOKEN, f'{DEFAULT_IMAGE_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_IMAGE_END_TOKEN}')
        conversation['output'] = conversation['output'].replace(DEFAULT_IMAGE_TOKEN, f'{DEFAULT_IMAGE_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_IMAGE_END_TOKEN}')

        return image, conversation, noise_image_index



class DataCollatorForCausalLM(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, source_max_len: int, target_max_len: int, train_on_source: bool, predict_with_generate: bool, apply_chat_template: bool = True):
        """
        Args:
            train_on_source (bool): whether use the source part, i.e., prompt or user input, as the training data.
        """
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.train_on_source = train_on_source
        self.predict_with_generate = predict_with_generate
        self.apply_chat_template = apply_chat_template

    def __call__(self, samples):
        images = [x[0] for x in samples]
        conversations = [x[1] for x in samples]
        noise_image_indices = [x[2] for x in samples]

        if self.apply_chat_template:
            conversations_standard = []
            for conv in conversations:
                new_conv = [
                    {"role": "user", "content": conv['input']},
                    {"role": "assistant", "content": conv['output']}
                ]
                conversations_standard.append(new_conv)
            conversations_standard = [self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in conversations_standard]

            sources = []
            targets = []
            
            # TODO: this is a hack, currently we only support these two chat templates
            if "<|user|>" in self.tokenizer.chat_template:
                conv_roles = ["<|user|>", "<|assistant|>"]
            elif "### Human:" in self.tokenizer.chat_template:
                conv_roles = ["### Human:", "### Assistant:"]
            
            for conv in conversations_standard:
                split_inds = []
                index = min(conv.find(conv_roles[1]), conv.find(conv_roles[0]))
                while index != -1:
                    split_inds.append(index)
                    index = conv.find(conv_roles[1], index + 1)
                split_inds.append(len(conv))
                split_inds[0] = 0
                assert (len(split_inds)-1) % 2 == 0, f"The number of roles should be even: {len(split_inds)}"
                for i in range(len(split_inds)-1):
                    if i % 2 == 0:
                        sources.append(conv[split_inds[i]:split_inds[i+1]])
                    else:
                        targets.append(conv[split_inds[i]:split_inds[i+1]])
        else:
            sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in conversations]
            targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in conversations]
        
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        
        # NOTE: this is a hack to substitue the <PAD> token after the image start token <|im_start|> with image token <image> back
        im_start_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_START_TOKEN)
        image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        for cur_input_ids in input_ids:
            # NOTE: make sure only one image exists
            for idx in torch.where(cur_input_ids == im_start_token_id):
                if len(idx) != 0 and cur_input_ids[idx + 1] == self.tokenizer.pad_token_id:
                    cur_input_ids[idx + 1] = image_token_id

        # Apply padding
        # TODO: we need to replace <image> token as the real image tokens (which may have different length),
        # so padding is required afterwards, maybe here we do not need to apply padding.
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels

        data_dict['images'] = images
        data_dict['noise_image_indices'] = noise_image_indices
        
        return data_dict


def get_train_sampler(dataset, rank, world_size, batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[
            (rank + offs) % world_size::world_size
        ]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[
            :sample_indices.size(0) - fill_ptr
        ]
        sample_indices[fill_ptr: fill_ptr + epoch_sample_indices.size(0)] = \
            epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * batch_size // world_size:].tolist()


@torch.no_grad()
# def update_ema(ema_model, model, decay=0.9999):
def update_ema(ema_model, model, decay=0.5):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(),
                      logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_fsdp(model, fsdp_strategy: str = 'fsdp', mixed_precision: str = 'fp32', grad_precision: str = 'fp32'):
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[fsdp_strategy],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[mixed_precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[grad_precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    return model


def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    """
    learning rate decay scheduler (cosine with warmup)
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def setup_tokenizer(model, tokenizer):
    """
    Add image generation tokens to the tokenizer. And resize the embedding layer of the model to match the tokenizer vocab size.
    """
    vocab = tokenizer.get_vocab()
    is_new_tokens_added = False
    for token in [DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN, DEFAULT_PAD_TOKEN]:
        if token not in vocab:
            is_new_tokens_added = True
    if is_new_tokens_added is False:
        logger.info('all special tokens are already added to tokenizer')
    else:
        tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN], special_tokens=True)
    
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def build_transforms(image_size: int):
    """
    Build the image transform for image generation. It consists of the following steps:
    1. center crop the image to square.
    2. normalize the image.

    Args:
        image_size (int): size of the image to be generated.
    
    Returns:
        torchvision.transforms.Compose: transforms for image generation.
    """
    crop_transform = transforms.Lambda(lambda pil_img: center_crop_arr(pil_img, image_size))
    image_transform = transforms.Compose([
        crop_transform,
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                             inplace=True),
    ])
    return image_transform


def main(args):    
    # setup distributed training
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    config = read_config_from_file(args.config_file)
    
    if dist.get_rank() == 0:
        print(config)
        json.dump(config, open(os.path.join(args.output_dir, 'cfg.json'), 'w'))

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)
    config = read_config_from_file(args.config_file)
    if dist.get_rank() == 0:
        json.dump(config, open(os.path.join(args.output_dir, 'cfg.json'), 'w'))

    # Set seed for reproducibility
    device = f'cuda:{rank % torch.cuda.device_count():d}'
    seed = args.seed * world_size + rank
    # FIXME: Be caution of the seed use in iterable dataset !!!
    torch.manual_seed(seed)


    #======================================
    # Setup logger and experimental tracker
    #======================================
    if rank == 0:
        logdir = os.path.join(args.logdir, os.path.basename(os.path.normpath(args.output_dir)))
        os.makedirs(logdir, exist_ok=True)
        logging.basicConfig(
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f"{logdir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    log_level = args.log_level
    logger.setLevel(log_level)

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))
    # Log on each process a small summary
    logger.warning(
        f"Process rank: {rank}, device: {device}, n_gpu: {world_size}"
        + f" distributed training: {bool(rank != -1)}, mixed precision: {args.mixed_precision}"
    )
    logger.info(f"Training configuration: " + json.dumps(config, indent=2))
    logger.info(f"Experiment directory: {args.output_dir}")

    if rank == 0:
        tracker_dir = os.path.join(logdir, datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname())
        os.makedirs(tracker_dir, exist_ok=True)
        tb_logger = SummaryWriter(tracker_dir)
    else:
        tb_logger = None


    #================
    # Load tokenizer
    #================
    tokenizer = AutoTokenizer.from_pretrained(config.llm_pretrained_path, add_bos_token=True, add_eos_token=True)

    #======================
    # Load pretrained model
    #======================
    # dump model config
    # TODO: remove unused configs
    model_config = AutoConfig.from_pretrained(config.llm_pretrained_path)
    model_config.patch_size = config.patch_size
    model_config.in_channels = config.in_channels
    model_config.learn_sigma = config.learn_sigma
    model_config.tokenizer_max_length = config.tokenizer_max_length
    model_config.tokenizer_padding_side = config.get('tokenizer_padding_side', 'right')
    model_config.use_flash_attn = config.use_flash_attn
    model_config.image_size = args.resolution
    model_config.use_pos_embed = config.use_pos_embed
    model_config.decoder_t_embed = config.decoder_t_embed
    model_config.use_adaln_final_layer = config.use_adaln_final_layer
    model_config.use_bi_attn_img_tokens = config.get('use_bi_attn_img_tokens', False)
    model_config.add_pos_embed_each_layer = config.get('add_pos_embed_each_layer', False)
    model_config.use_hybrid_attn_mask = config.get('use_hybrid_attn_mask', None)

    logger.info('Loading pretrained models...')
    model = MonoFormerForCausalLM.from_pretrained(config.llm_pretrained_path, config=model_config)
    model.initialize_weights()

    model, tokenizer = setup_tokenizer(model, tokenizer)
    # set up the index of image token in the config, so it can be used when load the pretrained model
    model.config.image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

    # initialize the ema model
    model_ema = MonoFormerForCausalLM.from_pretrained(config.llm_pretrained_path, config=model_config)
    model_ema, tokenizer = setup_tokenizer(model_ema, tokenizer)
    model_ema.config.image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)  # for loading checkpoint from ema model
    model_ema.load_state_dict(model.state_dict())
    model_ema.to(device)
    model_ema.requires_grad_(False)

    logger.info(f"DiT Parameters: {model.parameter_count():,}")

    #==================================================
    # Resume model weights from checkpoint if available
    #==================================================
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            # path = os.path.basename(args.resume_from_checkpoint)
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is not None:
            resume_path = os.path.join(args.output_dir, path)
            # if rank == 0:
            logger.info(f"Loading model from checkpoint '{resume_path}'.")
            model = model.from_pretrained(resume_path)
            ema_path = os.path.join(resume_path, "ema")
            logger.info(f"Loading ema model from checkpoint '{ema_path}'.")
            model_ema = model_ema.from_pretrained(ema_path)

    dist.barrier()

    # setup fsdp, model weights will broadcast from main process to all processes
    model = setup_fsdp(model, mixed_precision=args.mixed_precision)
    model_ema = setup_fsdp(model_ema, mixed_precision=args.mixed_precision)

    # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(config.vae_pretrained_path).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            # path = os.path.basename(args.resume_from_checkpoint)
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            logger.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            global_step = 0
        else:
            resume_path = os.path.join(args.output_dir, path)
            logger.info(f"Resuming optimizer states from: {resume_path}")
            try:
                optimizer_state_dict_path = os.path.join(resume_path, f"optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth")
                optimizer.load_state_dict(torch.load(optimizer_state_dict_path, map_location="cpu"))
            except Exception as e:
                logger.warning(f"Failed loading optimizer state: optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth, continue resume training without it.")
            
            # overwrite the optimizer parameters
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
                param_group["weight_decay"] = args.weight_decay
            dirname = os.path.basename(os.path.normpath(path))
            global_step = int(dirname.split("-")[1])

    else:
        global_step = 0

    #=============================
    # Build dataset and dataloader
    #=============================
    image_transform = build_transforms(args.resolution)  # TODO: chage 
    
    datasets = []
    data_weights = []
    for dataset_config in config.datasets:
        if dataset_config.type == 'ImageNetDataset':
            dataset = ImageNetDataset(
                data_root=dataset_config.data_root,
                item_processor=ImageNetItemProcessor(
                    image_transform,
                    args.text_dropout_prob,
                )
            )
        elif dataset_config.type == 'UltraChatDataset':
            dataset = UltraChatDataset(
                data_root=dataset_config.data_root,
                item_processor=UltraChatItemProcessor(),
                seed=args.seed,
                num_processes=dist.get_world_size(),
                process_rank=dist.get_rank(),
            )
        elif dataset_config.type == 'JourneyDBDataset':
            dataset = JourneyDBDataset(
                data_root=dataset_config.data_root,
                annotation_path=dataset_config.annotation_path,
                item_processor=JourneyDBItemProcessor(
                    image_transform,
                    args.text_dropout_prob,
                    dataset_config.prompt_type,
                ),
            )
        else:
            raise NotImplementedError(f'dataset type: {dataset_config.type} is not implemented.')
        
        datasets.append(dataset)
        data_weights.append(getattr(dataset_config, 'weight', 1))

    if config.use_iterable_dataset:
        # convert map-style dataset to iterable dataset
        for i, dataset in enumerate(datasets):
            if not isinstance(dataset, torchdata.IterableDataset):
                datasets[i] = ToIterableDataset(dataset)
        sampler = None
        dataset = MixIterDataset(datasets, data_weights)
    else:
        for dataset in datasets:
            if isinstance(dataset, torchdata.IterableDataset):
                raise RuntimeError(f'{dataset} is not map-style dataset, remove it or set `config.use_iterable_dataset` to True')
        dataset = ConcatDataset(datasets)
        global_batch_size = args.batch_size_per_gpu * dist.get_world_size() * args.gradient_accumulation_steps
        # TODO: map-style dataset does not support weighted sampling currently
        sampler = get_train_sampler(dataset, rank=dist.get_rank(), world_size=dist.get_world_size(), batch_size=global_batch_size, max_steps=args.max_steps, resume_step=global_step, seed=args.seed)

    use_chat_template = config.get('apply_chat_template', False)
    if use_chat_template:
        train_on_source = False
    else:
        train_on_source = True
    
    dataloader_collate_fn = DataCollatorForCausalLM(
        tokenizer=tokenizer, 
        source_max_len=config.source_max_length, 
        target_max_len=config.tokenizer_max_length-config.source_max_length, 
        train_on_source=train_on_source, 
        predict_with_generate=False, 
        apply_chat_template=use_chat_template
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataloader_collate_fn,
    )

    # Prepare models for training:
    # important! This enables embedding dropout for classifier-free guidance
    model.train()

    #=====================
    # Training Loop Starts
    #=====================
    start_time = time()
    train_losses = {
        'total_loss': 0.0,
        'diffusion_loss': 0.0,
        'lm_loss': 0.0,
    }
    if args.mixed_precision == "bf16":
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif args.mixed_precision == "fp16":
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    elif args.mixed_precision == "tf32" or args.precision == "fp32":
        autocast_ctx = contextlib.nullcontext()
    
    logger.info(f"Training for {args.max_steps:,} steps...")

    for step, data_dict in enumerate(loader, start=global_step):
        # apply lr schedule
        use_lr_decay = config.get('use_lr_decay', False)
        if use_lr_decay:
            learning_rate = args.lr
            warmup_iters = config.warmup_iters
            lr_decay_iters = config.lr_decay_iters
            min_lr = args.lr * config.lr_decay_rate
            lr = get_lr(step, warmup_iters, learning_rate, lr_decay_iters, min_lr) if use_lr_decay else learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        x = [img.to(device, non_blocking=True) for img in data_dict['images']]
        input_ids = data_dict['input_ids'].to(device)
        labels = data_dict['labels'].to(device)
        attention_mask = data_dict['attention_mask'].to(device)
        noise_image_indices = data_dict['noise_image_indices']

        with torch.no_grad():
            x_new = [None] * len(x)
            imgs = []
            img_inds = []
            for idx, img in enumerate(x):
                if img.shape[0] == 3:
                    imgs.append(img)
                    img_inds.append(idx)
                elif img.shape[0] == 8:
                    # load extracted vae features
                    img = DiagonalGaussianDistribution(img[None], True).sample().mul_(0.18215)[0].to(device)
                    x_new[idx] = img
                elif img.shape[0] == 0:
                    # no image
                    x_new[idx] = img
                else:
                    raise RuntimeError(f"Unexpected image shape: {img.shape}")
            # extract vae features
            if len(imgs) != 0:
                imgs = vae.encode(torch.stack(imgs)).latent_dist.sample().mul_(0.18215).to(device)
                for i, idx in enumerate(img_inds):
                    x_new[idx] = imgs[i]
            x = x_new

        t = torch.randint(0, diffusion.num_timesteps, (len(x),), device=device)
        # for image understanding task, where image is used for understanding rather than generaion, noise is not added
        for i, noise_image_index in enumerate(noise_image_indices):
            if noise_image_index == []:
                t[i] = 0

        noise = [torch.randn_like(img_start) for img_start in x]
        x_t = [diffusion.q_sample(x[i][None], t[i: i + 1], noise=noise[i][None])[0] for i in range(len(x))]

        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=x,
                noise_image_indices=noise_image_indices,
                t=t,
                x_t=x_t,
            )
        
        x_indices = outputs['x_indices']  # batch indices
        if len(x_indices) != 0:
            model_output = outputs['x_out']
            x_start = [x[i] for i in x_indices]
            for out, x0 in zip(model_output, x_start):
                if out.shape[-2:] != x0.shape[-2:]:
                    raise
            x_t = [x_t[i] for i in x_indices]
            noise = [noise[i] for i in x_indices]
            t = torch.stack([t[i] for i in x_indices])
            diffusion_terms = diffusion.training_losses(model_output=model_output, x_t=x_t, x_start=x_start, t=t, noise=noise)
        else:
            diffusion_terms = {'loss': torch.tensor(0., device=device)}
        
        diffusion_loss = (diffusion_terms["loss"].mean() * config.loss_dict['loss_diffusion']) / args.gradient_accumulation_steps
        lm_loss = (outputs['loss'] * config.loss_dict['loss_lm']) / args.gradient_accumulation_steps

        loss = diffusion_loss + lm_loss
        loss.backward()
        
        train_losses['total_loss'] += loss.item()
        train_losses['diffusion_loss'] += diffusion_loss.item()
        train_losses['lm_loss'] += lm_loss.item()
        
        # gather and average loss across all processes for logging
        for key in train_losses:
            loss_tensor = torch.tensor(train_losses[key]).to(device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= world_size
            train_losses[key] = loss_tensor.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            global_step += 1
            
            # apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            
            update_ema(model_ema, model)
            
            if tb_logger is not None:
                tb_logger.add_scalar("train/loss", loss.item(), global_step)
                tb_logger.add_scalar("train/lm_loss", lm_loss.item(), global_step)
                tb_logger.add_scalar("train/diffusion_loss", diffusion_loss.item(), global_step)
                tb_logger.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            
            # print log every log_every steps
            if global_step % args.log_steps == 0:
                end_time = time()
                secs_per_step = (end_time - start_time) / args.log_steps
                imgs_per_sec = args.batch_size_per_gpu * world_size * args.gradient_accumulation_steps * args.log_steps / (
                    end_time - start_time
                )
                # TODO: Reduce loss history over all processes:
                for key in train_losses.keys():
                    train_losses[key] /= args.log_steps

                logger.info(f"(step={global_step:07d}) "
                            f"Train Loss: {train_losses['total_loss']:.4f}, "
                            f"Diffusion Loss: {train_losses['diffusion_loss']:.4f}, "
                            f"LM Loss: {train_losses['lm_loss']:.4f}, "
                            f"Train Secs/Step: {secs_per_step:.2f}, "
                            f"Train Imgs/Sec: {imgs_per_sec:.2f}")
                start_time = time()

                for key in train_losses.keys():
                    train_losses[key] = 0.0
            
            if (
                (global_step) % args.checkpointing_steps == 0
                or (global_step) == args.max_steps
            ):
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at most `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                # save the pretrained model only on the main process
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    logger.info(f"Saved model to {save_path}.")
                    model_state_dict = model.state_dict()
                    if dist.get_rank() == 0:
                        model.save_pretrained(save_path, is_main_process=(dist.get_rank() == 0), safe_serialization=False, state_dict=model_state_dict)

                model_ema_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}", 'ema')
                with FSDP.state_dict_type(
                    model_ema,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    logger.info(f"Saved EMA model to {model_ema_save_path}.")
                    model_ema_state_dict = model_ema.state_dict()
                    if dist.get_rank() == 0:
                        model_ema.save_pretrained(model_ema_save_path, is_main_process=(dist.get_rank() == 0), safe_serialization=False, state_dict=model_ema_state_dict)

                # save the optimizer state dict
                with FSDP.state_dict_type(
                    model,
                    StateDictType.LOCAL_STATE_DICT,
                ):
                    optimizer_save_path = os.path.join(save_path, f"optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth")
                    logger.info(f"Saved optimizer to {optimizer_save_path}.")
                    torch.save(optimizer.state_dict(), optimizer_save_path)

                if dist.get_rank() == 0:
                    tokenizer.save_pretrained(save_path)
                    torch.save(args, os.path.join(save_path, "model_args.pth"))
                    json.dump(config, open(os.path.join(save_path, 'cfg.json'), 'w'))

    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--max_steps", type=int, default=100_000,
        help="Number of training steps."
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"],
                        default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision",
                        choices=["fp32", "tf32", "fp16", "bf16"],
                        default="bf16")
    parser.add_argument("--grad_precision",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=2.0,
        help="Clip the L2 norm of the gradients to the given value."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--text_dropout_prob",
        type=float,
        default=0.1,
        help="Randomly change the caption of a sample to a blank string with the given probability."
    )
    parser.add_argument(
        "--config_file", type=str,
    )
    parser.add_argument(
        "--seed", type=int, default=0,
    )
    parser.add_argument(
        "--allow_tf32", action="store_true"
    )
    parser.add_argument(
        "--resolution", type=int, default=256
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        '--checkpoints_total_limit',
        type=int,
        default=None,
        help='Maximum number of checkpoints to keep before deleting old ones. If set to None, no limit is enforced.'
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps."
    )
    parser.add_argument("--batch_size_per_gpu", type=int, default=16, help="The global batch size is equal to (batch_size_per_gpu * n_gpus * gradient_accumulation_steps)")
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--checkpointing_steps", type=int, default=50_000)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument(
        "--logdir", type=str, default="logs"
    )
    args = parser.parse_args()

    main(args)

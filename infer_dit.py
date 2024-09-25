import os
import json
import argparse
from typing import List

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import to_pil_image
import transformers
from transformers import AutoTokenizer
from diffusers.models import AutoencoderKL

from config import read_config_from_file
from constants import DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN, DEFAULT_PAD_TOKEN, DEFAULT_IMAGE_TOKEN
from diffusion import create_diffusion
from models import MonoFormerForCausalLM


def preprocess_single_inputs(tokenizer: transformers.PreTrainedTokenizer, inputs: List[str], max_length=512, device='cuda'):
    """
    Steps to preprocess inputs:
    1. add special geenration tokens after inputs: <|im_start|><image><|im_end|>
    2. add bos token before inputs
    3. tokenize inputs
    4. replace tokens after bos and before <|im_start|> with padding tokens to form unconditional inputs
    5. concatenate conditional inputs and unconditional inputs along batch dimension
    6. create attention masks by masking padding tokens
    7. create noise image indices
    """
    input_ids = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
        # return_tensors="pt",
    )['input_ids']
    
    input_ids = [torch.tensor(i) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    # FIXME: replace pad token after <|im_start|> with <image>, this is due to tokenizer cannot correctly tokenize <image> after <|im_start|>
    im_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_START_TOKEN)
    im_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_END_TOKEN)
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    for cur_input_ids in input_ids:
        for idx in torch.where(cur_input_ids == im_start_token_id):
            if cur_input_ids[idx + 1] == tokenizer.pad_token_id:
                cur_input_ids[idx + 1] = image_token_id

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    noise_image_indices = [[0] for _ in range(len(input_ids))]
 
    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'noise_image_indices': noise_image_indices,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--vae_pretrained_path", type=str, default='stabilityai/sd-vae-ft-mse')
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512, 1024])
    args = parser.parse_args()

    assert args.num_gpus == 1, "Currently only support inference with 1 gpu."

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    cfg = read_config_from_file(os.path.join(args.ckpt, 'cfg.json'))
    if rank == 0:
        print('========= Training config: =========')
        print(json.dumps(cfg, indent=4))

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    
    # load model
    if args.ema:
        model = MonoFormerForCausalLM.from_pretrained(os.path.join(args.ckpt, 'ema'))
    else:
        model = MonoFormerForCausalLM.from_pretrained(args.ckpt)
    model.eval().cuda()
    
    # load vae
    vae = AutoencoderKL.from_pretrained(
        args.vae_pretrained_path,
    ).cuda()

    # create diffusion
    num_sampling_steps = 20
    w = 256
    h = 256
    cfg_scale = 4
    diffusion = create_diffusion(str(num_sampling_steps))
    latent_w, latent_h = w // 8, h // 8
    z = torch.randn([1, 4, latent_w, latent_h], device="cuda")
    z = z.repeat(2, 1, 1, 1)

    use_chat_template = cfg.get('use_chat_template', False)
    if use_chat_template:
        DEFAULT_CHAT_TEMPLATE = "{%- for message in messages %}{%- if message['role'] == 'user' %}{{- bos_token + '### Human: ' + message['content'].strip() }}{%- elif message['role'] == 'assistant' %}{{- '### Assistant: ' + message['content'] + eos_token }}{%- endif %}{%- if loop.last and add_generation_prompt %}{{- '### Assistant: '}}{%- endif %}{%- endfor %}"
        tokenizer.chat_template = cfg.get('chat_template', DEFAULT_CHAT_TEMPLATE)
    
    # begin generation
    prompt = "Please generate an image of beach wagon"

    if use_chat_template:
        conversations = [
            [{"role": "user", "content": f"{prompt}"}, {"role": "assistant", "content": f"{DEFAULT_IMAGE_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_IMAGE_END_TOKEN}"}],
            [{"role": "user", "content": ""}, {"role": "assistant", "content": f"{DEFAULT_IMAGE_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_IMAGE_END_TOKEN}"}],
        ]
        inputs = [
            tokenizer.apply_chat_template(conversations[0], tokenize=False, add_generation_prompt=False),
            tokenizer.apply_chat_template(conversations[1], tokenize=False, add_generation_prompt=False)
        ]
    else:
        inputs = [
            f"{tokenizer.bos_token}{prompt}{DEFAULT_IMAGE_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_IMAGE_END_TOKEN}",
            f"{tokenizer.bos_token}{DEFAULT_IMAGE_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_IMAGE_END_TOKEN}"
        ]

    inputs_dict = preprocess_single_inputs(tokenizer, inputs)
    inputs_dict['cfg_scale'] = cfg_scale

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=inputs_dict, progress=dist.get_rank() == 0, device="cuda",
    )

    samples = samples[:1]

    samples = vae.decode(samples.float() / 0.18215).sample
    samples = (samples + 1.) / 2.
    samples.clamp_(0., 1.)
    img = to_pil_image(samples[0])
    
    img.save(f"output_image.png")
    
    # modify the prompt and copy-paste the above code for another generation
    from IPython import embed; embed()
    

if __name__ == "__main__":
    main()

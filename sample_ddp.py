# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import json
import torch
import torch.distributed as dist
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from config import read_config_from_file
from transformers import AutoTokenizer
from MonoFormer.models.monoformer import LlamaDitForCausalLM
from constants import DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN, DEFAULT_PAD_TOKEN, DEFAULT_IMAGE_TOKEN
import transformers
from typing import List
from torch.nn.utils.rnn import pad_sequence


def preprocess_single_inputs(tokenizer: transformers.PreTrainedTokenizer, inputs: List[str], max_length=512, use_chat_template=False, device='cuda'):
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
    generation_tokens = f"{DEFAULT_IMAGE_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_IMAGE_END_TOKEN}"
    # inputs = [f"{tokenizer.bos_token}{example}{generation_tokens}" for example in inputs]
    # if use_chat_template:
    #     inputs = [tokenizer.apply_chat_template(example, tokenize=False, add_generation_prompt=False) for example in inputs]
    # else:
    inputs = [f"{tokenizer.bos_token}{example}{generation_tokens}" for example in inputs]
    
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


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    fs_init.initialize_model_parallel(dist.get_world_size())
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    latent_size = args.image_size // 8
    
    cfg = read_config_from_file(os.path.join(args.ckpt, 'cfg.json'))
    if dist.get_rank() == 0:
        print('========== Training configuration: ==========')
        print(json.dumps(cfg, indent=2))

    if args.ema:
        model = LlamaDitForCausalLM.from_pretrained(os.path.join(args.ckpt, 'ema'), torch_dtype=torch.float32)
    else:
        model = LlamaDitForCausalLM.from_pretrained(args.ckpt)
    
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    # tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    
    model.eval().cuda()

    vae = AutoencoderKL.from_pretrained(
        args.vae_pretrained_path,
    ).cuda()

    diffusion = create_diffusion(str(args.num_sampling_steps))

    labels = json.load(open(args.imagenet_labels_path, 'r'))
    labels = list(labels.values())
    
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    sample_folder_dir = args.sample_dir
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        indices = torch.randint(0, args.num_classes, (n,))
        # torch.randint(0, num_classes, (20,))
        # y = [labels[i]['label'] for i in indices]
        y = [f"Please generate an image of {labels[i]['label']}." for i in indices]
        
        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            # y_null = torch.tensor([1000] * n, device=device)
            y_null = [""] * len(y)
            # y = torch.cat([y, y_null], 0)
            y = y + y_null
            # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            # sample_fn = model.forward_with_cfg

            inputs_dict = preprocess_single_inputs(tokenizer, y)
            inputs_dict['cfg_scale'] = args.cfg_scale
            sample_fn = model.forward_with_cfg
        else:
            # model_kwargs = dict(y=y)
            # sample_fn = model.forward
            raise NotImplementedError("Currently only support classifier-free guidance")
        
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False,
            model_kwargs=inputs_dict, progress=False, device="cuda",
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
    
    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--imagenet_labels_path", type=str)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--vae_pretrained_path", type=str)
    args = parser.parse_args()
    main(args)

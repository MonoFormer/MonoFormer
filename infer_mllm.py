import argparse
import json
import multiprocessing as mp
import os
import socket
from typing import List, Optional

from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init

import transformers
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
from transformers import pipeline

from PIL import Image
from diffusion import create_diffusion
import models

from models import MonoFormerForCausalLM
from config import read_config_from_file
from torchvision import transforms
from imgproc import resize_and_pad_image

from constants import DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN, DEFAULT_PAD_TOKEN, DEFAULT_IMAGE_TOKEN


def preprocess_inputs(tokenizer: transformers.PreTrainedTokenizer, inputs: List[str], images: List[torch.Tensor], max_length=512, device='cuda'):
    """
    Currently, only support batch size 1.
    """
    assert len(inputs) == 1

    input_ids, attention_mask = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).values()
    
    if len(images) > 0:
        # FIXME: replace pad token after <|im_start|> with <image>, this is due to tokenizer cannot correctly tokenize <image> after <|im_start|>
        im_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_START_TOKEN)
        im_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_END_TOKEN)
        image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        for cur_input_ids in input_ids:
            for idx in torch.where(cur_input_ids == im_start_token_id):
                if cur_input_ids[idx + 1] == tokenizer.pad_token_id:
                    cur_input_ids[idx + 1] = image_token_id

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        flags = [[1]]
    else:
        flags = []

    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'images': [images],
        'flags': flags,
        't': torch.tensor([0]).to(device),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512, 1024])
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # fs_init.initialize_model_parallel(args.num_gpus)
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

    use_chat_template = cfg.get('use_chat_template', False)
    tokenizer.chat_template = cfg.chat_template if use_chat_template else None

    def prepare_inputs(prompt, image_path=None, resolution=256, device='cuda'):
        if image_path is not None:
            images = [Image.open(image_path).convert("RGB")]
            prompt = f"{DEFAULT_IMAGE_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_IMAGE_END_TOKEN}\n{prompt}"
        else:
            images = []

        return prompt, images

    prompt = "Can you describe the image?"
    image_path = "/root/paddlejob/workspace/box5_home/disk1/vis/chuyang/LLaMA2-Accessory-main/LlamaDiT/outputs/output_200k_laion_3_init_imagenet.png"
    # prompt = "Can you briefly introduce Berlin?"
    # image_path = None

    prompt, images = prepare_inputs(prompt, image_path, args.resolution)

    if use_chat_template:
        conversation = [{"role": "user", "content": prompt,}]
        inputs = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)]
    else:
        inputs = [f"{tokenizer.bos_token}{prompt}"]

    inputs_dict = preprocess_inputs(tokenizer, inputs, images)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs_dict['input_ids'],
            attention_mask=inputs_dict['attention_mask'],
            flags=inputs_dict['flags'],
            images=inputs_dict['images'],
            t=inputs_dict['t'],
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=256,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
    print(outputs)


if __name__ == "__main__":
    main()

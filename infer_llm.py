import argparse
import json
import os
from typing import List

import transformers
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from models import MonoFormerForCausalLM
from config import read_config_from_file


def preprocess_inputs(tokenizer: transformers.PreTrainedTokenizer, inputs: List[str], max_length=512, device='cuda'):
    """
    Currently, only support batch size 1.
    """
    # inputs = [f"{tokenizer.bos_token}{example}" for example in inputs]
    input_ids, attention_mask = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).values()

    noise_image_indices = [None]

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
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512, 1024])
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

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
    
    model.forward = model.forward_wo_image

    use_chat_template = cfg.get('use_chat_template', False)
    if use_chat_template:
        DEFAULT_CHAT_TEMPLATE = "{%- for message in messages %}{%- if message['role'] == 'user' %}{{- bos_token + '### Human: ' + message['content'].strip() }}{%- elif message['role'] == 'assistant' %}{{- '### Assistant: ' + message['content'] + eos_token }}{%- endif %}{%- if loop.last and add_generation_prompt %}{{- '### Assistant: '}}{%- endif %}{%- endfor %}"
        tokenizer.chat_template = cfg.get('chat_template', DEFAULT_CHAT_TEMPLATE)

    # begin generation
    prompt = "### Human: Can you briefly introduce Berlin? ### Assistant:"

    if use_chat_template:
        conversation = [{"role": "user", "content": prompt,}]
        inputs = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)]
    else:
        inputs = [f"{tokenizer.bos_token}{prompt}"]

    inputs_dict = preprocess_inputs(tokenizer, inputs)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs_dict['input_ids'],
            attention_mask=inputs_dict['attention_mask'],
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
    
    # modify the prompt and copy-paste the above code for another generation
    from IPython import embed; embed()


if __name__ == "__main__":
    main()

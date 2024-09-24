llm_pretrained_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
vae_pretrained_path = 'stabilityai/sd-vae-ft-mse'
# llm_pretrained_path = '/root/paddlejob/workspace/box5_home/disk1/vis/songyuxin02/data/checkpoints/TinyLlama-1.1B-Chat-v1.0'
# vae_pretrained_path = '/root/paddlejob/workspace/box5_home/disk1/vis/songyuxin02/data/checkpoints/sd-vae-ft-ema'

# dataset settings
use_iterable_dataset = False
datasets = [
    dict(type='ImageNetDataset', data_root='/root/paddlejob/workspace/imagenet/imagenet/train', weight=1),
]

# model settings
patch_size = 2
in_channels = 4
learn_sigma = True
use_flash_attn = False
use_pos_embed = True
decoder_t_embed = "add_before_image_tokens"  # choose from ['add_before_decoder', 'add_at_each_layer', 'add_with_embed_at_each_layer', 'adaln_at_each_layer', 'adaln_before_decoder', 'add_before_image_tokens']
use_adaln_final_layer = True
use_bi_attn_img_tokens = True

# tokenizer settings
tokenizer_max_length = 512
source_max_length = 128
use_chat_template = False
chat_template = "{%- for message in messages %}{%- if message['role'] == 'user' %}{{- bos_token + '### Human: ' + message['content'].strip() }}{%- elif message['role'] == 'assistant' %}{{- '### Assistant: ' + message['content'] + eos_token }}{%- endif %}{%- if loop.last and add_generation_prompt %}{{- '### Assistant: '}}{%- endif %}{%- endfor %}"

# optimizer settings
use_lr_decay = False
# only valid when lr_decay is set to True
warmup_iters = 500
lr_decay_iters = 300000
lr_decay_rate = 0.1

loss_dict = {
    'loss_lm': 0.01,
    'loss_diffusion': 1.0
}

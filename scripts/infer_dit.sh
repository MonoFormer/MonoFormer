ckpt=/root/paddlejob/workspace/monoformer_imagenet_res256_bf16_bs32_lr1e-4
vae_pretrained_path=stabilityai/sd-vae-ft-mse
CUDA_VISIBLE_DEVICES=7 torchrun --master_port 39500 --nproc_per_node 1 infer_dit.py --ckpt $ckpt --resolution 256 --vae_pretrained_path $vae_pretrained_path --ema

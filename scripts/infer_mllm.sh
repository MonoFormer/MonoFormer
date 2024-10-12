ckpt=path/to/checkpoint
CUDA_VISIBLE_DEVICES=1 torchrun --master_port 39500 --nproc_per_node 1 infer_mllm.py --ckpt $ckpt

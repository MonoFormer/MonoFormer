ckpt=...
CUDA_VISIBLE_DEVICES=1 torchrun --master_port 39500 --nproc_per_node 1 infer_llm.py --ckpt $ckpt

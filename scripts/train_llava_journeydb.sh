#!/usr/bin/env sh
gradient_accumulation_steps=2
batch_size_per_gpu=4
lr=1e-4
precision=bf16
resolution=256
batch_size=$((batch_size_per_gpu * gradient_accumulation_steps))

exp_name=monoformer_llava_journeydb_res${resolution}_${precision}_bs${batch_size}_lr${lr}
mkdir -p /root/paddlejob/workspace/results/${exp_name}


torchrun --nproc_per_node 2 train.py \
    --output_dir /root/paddlejob/workspace/results/${exp_name} \
    --lr ${lr} \
    --batch_size_per_gpu $batch_size_per_gpu \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_grad_norm 2.0 \
    --max_steps 500000 \
    --checkpointing_steps 10 --log_steps 100 \
    --mixed_precision $precision --grad_precision fp32 \
    --resolution $resolution \
    --config_file configs/train_llava_journeydb.py

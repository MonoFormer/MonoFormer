sample_dir=...
ckpt_dir=...

torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py --ckpt $ckpt_dir \
    --per-proc-batch-size 64 \
    --resolution 256 \
    --ema \
    --num-fid-samples 10000 \
    --imagenet_labels_path data/imagenet_labels.json \
    --cfg-scale 4 \
    --num-sampling-steps 20 \
    --sample-dir $sample_dir \

steps=20
cfg=4
sample_dir=samples/monoformer_step_${steps}_cfg_${cfg}
ckpt_dir=...
vae_pretrained_path='stabilityai/sd-vae-ft-mse'

torchrun --nnodes=1 --nproc_per_node=8 sample_t2i.py --ckpt $ckpt_dir \
    --per-proc-batch-size 64 \
    --resolution 256 \
    --ema \
    --num-fid-samples 2000 \
    --cfg-scale $cfg \
    --num-sampling-steps $steps \
    --sample-dir $sample_dir \
    --vae_pretrained_path $vae_pretrained_path \
    --prompts-path data/parti_prompts.json

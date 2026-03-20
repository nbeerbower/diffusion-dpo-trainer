#!/bin/bash
# Example training launch script for DPO diffusion training.
# Adjust paths, dataset names, and hyperparameters for your use case.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python train.py \
    --model_path /path/to/your/model.safetensors \
    --dataset "your-hf-username/your-dpo-dataset" \
    --dataset_split "train" \
    --shuffle_dataset \
    --base_model stabilityai/stable-diffusion-xl-base-1.0 \
    --output_dir ./output \
    --num_epochs 10 \
    --batch_size 1 \
    --use_adafactor \
    --learning_rate 2e-6 \
    --lr_scheduler cosine \
    --lr_warmup_steps 250 \
    --sft_weight 0.3 \
    --beta 0.4 \
    --beta_schedule linear \
    --beta_warmup_steps 500 \
    --gradient_accumulation_steps 16 \
    --grad_clip_norm 0.3 \
    --logit_clamp 10 \
    --save_steps 50 \
    --image_size 1024 \
    --enable_gradient_checkpointing \
    --enable_xformers \
    --freeze_unet_strategy color_blocks \
    --freeze_unet_layers "0,1" \
    --mixed_precision fp16 \
    --use_base_vae \
    --use_wandb \
    --wandb_project "my-dpo-project" \
    --wandb_run_name "experiment-1"

# CLAUDE.md

## Project overview

DPO (Direct Preference Optimization) trainer for Stable Diffusion XL. Trains the UNet to prefer chosen images over rejected images using a DPO loss with SFT regularization.

## Architecture

- `train.py` - Entry point. Handles arg parsing, model loading, dataset loading, training loop, checkpointing, and W&B logging. Calls into the `dpo/` package for core logic.
- `inference.py` - Standalone script for generating images from a trained model directory.
- `dpo/dataset.py` - `DPODataset` wraps a HuggingFace dataset. Tokenizes prompts with both SDXL tokenizers, resizes images to `image_size`, normalizes to [-1, 1].
- `dpo/trainer.py` - `DPOTrainer` handles the DPO+SFT loss computation, optimizer setup, LR/beta scheduling, and gradient accumulation. Only the UNet is trained; text encoders and VAE are frozen.
- `dpo/plotting.py` - `create_loss_plots()` generates matplotlib loss visualizations from training history.

## Key design decisions

- **VAE is always float32** regardless of mixed precision setting. This prevents NaN in latent encoding. The VAE is tested at startup and auto-reloaded from the base model if it produces NaN.
- **Timestep sampling is biased to [30%, 80%]** of the scheduler range rather than uniform. This avoids very noisy or nearly clean timesteps where the DPO signal is weak.
- **DPO log-ratios are clamped** via `--logit_clamp` to prevent extreme gradient spikes.
- **Text encoders are not prepared with accelerator** - they're frozen and moved to device manually. Only the UNet and dataloader go through `accelerator.prepare()`.
- **The optimizer is not prepared with accelerator** either, to avoid FP16 gradient issues with the DPO loss computation.
- **Model weight loading** auto-detects whether the safetensors file is a full checkpoint (with `unet.`/`vae.` prefixed keys) or UNet-only weights.

## Common tasks

### Run training
```bash
bash example_train.sh
# or directly:
python train.py --model_path model.safetensors --dataset "user/dataset" --output_dir ./output
```

### Run inference
```bash
python inference.py --model_path ./output/final_model --prompt "your prompt"
```

### Run with multiple datasets
```bash
python train.py --dataset "user/ds1" "user/ds2" --shuffle_dataset
```

## Dependencies

Core: `torch`, `diffusers`, `transformers`, `accelerate`, `datasets`, `safetensors`, `Pillow`, `numpy`, `tqdm`

Optional: `wandb`, `bitsandbytes`, `xformers`, `matplotlib`

## Style

- Python files use double quotes for strings
- No type annotations beyond what's naturally readable
- Minimal comments - code should be self-explanatory
- Args use snake_case with `--` prefix

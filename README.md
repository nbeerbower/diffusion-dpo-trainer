# diffusion-dpo-trainer

DPO (Direct Preference Optimization) training for Stable Diffusion XL models. Trains the UNet to prefer chosen images over rejected images for a given prompt, with SFT regularization to prevent catastrophic forgetting.

## How it works

Given a dataset of `(prompt, chosen_image, rejected_image)` triples, the trainer:

1. Encodes both images to latent space via the frozen VAE
2. Adds shared noise at a shared timestep to both latents
3. Predicts noise with the UNet for both
4. Computes per-sample MSE losses against the true noise
5. Applies the DPO objective: `loss = -log_sigmoid(beta * (loss_rejected - loss_chosen))`
6. Adds an SFT term on chosen examples for regularization: `total = dpo_loss + sft_weight * sft_loss`

The model learns that the chosen image is a better match for the prompt by making the UNet predict noise more accurately for chosen images.

## Setup

```bash
pip install -r requirements.txt
```

Optional dependencies for extra features:
- `wandb` - experiment tracking
- `bitsandbytes` - 8-bit Adam optimizer
- `xformers` - memory-efficient attention
- `matplotlib` - loss plots

## Quick start

```bash
python train.py \
    --model_path /path/to/model.safetensors \
    --dataset "your-username/your-dpo-dataset" \
    --output_dir ./output \
    --num_epochs 10 \
    --batch_size 1 \
    --learning_rate 2e-6 \
    --beta 0.4 \
    --enable_gradient_checkpointing \
    --mixed_precision fp16
```

See `example_train.sh` for a full example with all the bells and whistles.

## Dataset format

Your HuggingFace dataset must have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | `str` | Text prompt describing the image |
| `chosen` | `PIL.Image` | Preferred image for this prompt |
| `rejected` | `PIL.Image` | Dispreferred image for this prompt |

Multiple datasets can be passed and will be auto-concatenated:

```bash
--dataset "user/dataset-a" "user/dataset-b" --shuffle_dataset
```

## Model loading

The `--model_path` accepts a `.safetensors` file that can be either:

- **Full checkpoint** (keys prefixed with `unet.`, `vae.`, `text_encoder.`, etc.) - UNet and VAE weights are extracted automatically
- **UNet-only** weights - loaded directly into the SDXL UNet

The base pipeline (text encoders, tokenizers, scheduler) is always loaded from `--base_model`. Use `--use_base_vae` to skip loading VAE weights from the safetensors file.

## Key arguments

### DPO hyperparameters

| Arg | Default | Description |
|-----|---------|-------------|
| `--beta` | `0.1` | DPO preference strength. 0.1-1 = standard, 1-10 = strong |
| `--sft_weight` | `0.1` | Weight for SFT regularization loss. Higher = more conservative |
| `--logit_clamp` | `5.0` | Clamps DPO log-ratios to prevent instability |
| `--beta_schedule` | `constant` | `constant`, `linear` (warmup + tail decay), or `cosine` |
| `--beta_warmup_steps` | `100` | Steps to ramp beta from 0 to target (linear/cosine schedules) |

### Optimizer

| Arg | Default | Description |
|-----|---------|-------------|
| `--learning_rate` | `5e-6` | Peak learning rate |
| `--lr_scheduler` | `constant` | `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `exponential` |
| `--lr_warmup_steps` | `100` | Warmup steps before peak LR |
| `--gradient_accumulation_steps` | `4` | Effective batch size multiplier |
| `--grad_clip_norm` | `1.0` | Max gradient norm |
| `--use_adafactor` | off | Use Adafactor instead of AdamW (lower memory) |
| `--use_8bit_adam` | off | Use 8-bit Adam (requires bitsandbytes) |

### Memory optimization

| Arg | Description |
|-----|-------------|
| `--mixed_precision fp16` | FP16 mixed precision (recommended) |
| `--enable_gradient_checkpointing` | Trade compute for memory |
| `--enable_xformers` | Memory-efficient attention (requires xformers) |

### UNet freezing

Freeze early UNet layers to focus training on higher-level features and reduce memory:

| Strategy | What it freezes |
|----------|----------------|
| `none` | Nothing (default) |
| `input_blocks` | Specified `--freeze_unet_layers` down blocks |
| `early_blocks` | Same as input_blocks |
| `color_blocks` | Specified down blocks + `conv_in` (initial pixel projection) |

```bash
--freeze_unet_strategy color_blocks --freeze_unet_layers "0,1"
```

### Weights & Biases

```bash
--use_wandb \
--wandb_project "my-project" \
--wandb_run_name "experiment-1" \
--wandb_tags dpo sdxl
```

Logs step-level losses, gradient norms, learning rate, beta values, and epoch summaries.

## Inference

Test a trained model:

```bash
python inference.py \
    --model_path ./output/final_model \
    --prompt "a cat sitting on a windowsill, oil painting" \
    --num_images 4 \
    --seed 42
```

## Outputs

After training, `--output_dir` will contain:

```
output/
  checkpoint-{step}/     # Periodic checkpoints (accelerator state + UNet)
  final_model/           # Full diffusers pipeline (load with from_pretrained)
  training_history.json  # Per-step and per-epoch loss data
  loss_plots.png         # Detailed 4-panel loss visualization
  epoch_losses.png       # Simple epoch-level loss plot
```

The `final_model/` directory can be loaded directly:

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained("./output/final_model")
```

## Project structure

```
diffusion-dpo-trainer/
  dpo/
    __init__.py      # Package exports
    dataset.py       # DPODataset - loads prompt/chosen/rejected triples
    trainer.py       # DPOTrainer - loss computation, scheduling, optimization
    plotting.py      # Training loss visualization
  train.py           # Main training entry point
  inference.py       # Generate images from a trained model
  example_train.sh   # Example launch script
  requirements.txt
```

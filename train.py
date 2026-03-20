#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) training for Stable Diffusion XL.

Trains an SDXL UNet to prefer chosen images over rejected images using the
DPO objective, with optional SFT regularization.

Supports:
- Custom safetensors model weights (full checkpoint or UNet-only)
- Multiple HuggingFace datasets (auto-concatenated)
- Fractional epochs
- LR scheduling (constant, linear, cosine, cosine_with_restarts, polynomial, exponential)
- Beta scheduling (constant, linear, cosine)
- Optimizers: AdamW, 8-bit Adam, Adafactor
- UNet layer freezing strategies
- Gradient checkpointing, xformers, VAE slicing/tiling
- Weights & Biases logging with sample image generation
"""

import os
import json
import argparse
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from diffusers import (
    StableDiffusionXLPipeline,
    DDPMScheduler,
    AutoencoderKL,
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from safetensors.torch import load_file
from tqdm import tqdm

from dpo import DPODataset, DPOTrainer, create_loss_plots

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Diffusion Training for SDXL")

    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to safetensors model file (full checkpoint or UNet-only)")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Base SDXL model for pipeline components")
    parser.add_argument("--use_base_vae", action="store_true",
                        help="Ignore VAE weights from model_path, use base model VAE instead")

    # Dataset
    parser.add_argument("--dataset", type=str, nargs="+", required=True,
                        help="HuggingFace dataset(s) for DPO training (multiple will be concatenated)")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--shuffle_dataset", action="store_true")
    parser.add_argument("--dataset_seed", type=int, default=None,
                        help="Seed for dataset shuffling (uses --seed if not specified)")

    # Training
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--num_epochs", type=float, default=10,
                        help="Number of epochs (supports decimals, e.g. 0.5, 1.5)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=500)

    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (preference strength: 0.1-1 standard, 1-10 strong)")
    parser.add_argument("--sft_weight", type=float, default=0.1,
                        help="SFT regularization weight (0.0-1.0)")
    parser.add_argument("--logit_clamp", type=float, default=5.0,
                        help="Clamp for DPO logit ratios")
    parser.add_argument("--beta_schedule", type=str, default="constant",
                        choices=["constant", "linear", "cosine"])
    parser.add_argument("--beta_warmup_steps", type=int, default=100)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["constant", "linear", "cosine", "cosine_with_restarts",
                                 "polynomial", "exponential"])
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_min", type=float, default=1e-9)
    parser.add_argument("--lr_cycles", type=int, default=1,
                        help="Cycles for cosine_with_restarts scheduler")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--use_adafactor", action="store_true")
    parser.add_argument("--adafactor_scale_parameter", action="store_true")
    parser.add_argument("--adafactor_relative_step", action="store_true")
    parser.add_argument("--adafactor_warmup_init", action="store_true")

    # Memory optimization
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")

    # UNet freezing
    parser.add_argument("--freeze_unet_strategy", type=str, default="none",
                        choices=["none", "input_blocks", "early_blocks", "color_blocks"])
    parser.add_argument("--freeze_unet_layers", type=str, default="0,1",
                        help="Comma-separated down_block indices to freeze")

    # Weights & Biases
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpo-sdxl-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None)

    # Debug
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def load_model_weights(pipe, model_path, use_base_vae, accelerator):
    """Load UNet (and optionally VAE) weights from a safetensors file."""
    state_dict = load_file(model_path)

    has_vae = any(k.startswith("vae.") for k in state_dict)
    has_unet = any(
        k.startswith("unet.") or not any(k.startswith(p) for p in ["text_encoder.", "text_encoder_2.", "vae."])
        for k in state_dict
    )

    if accelerator.is_main_process:
        print(f"Safetensors contents: VAE={has_vae}, UNet={has_unet}")

    if has_unet:
        unet_sd = {
            k: v for k, v in state_dict.items()
            if k.startswith("unet.") or not any(k.startswith(p) for p in ["text_encoder.", "text_encoder_2.", "vae."])
        }
        unet_sd = {k.replace("unet.", ""): v for k, v in unet_sd.items()}
        pipe.unet.load_state_dict(unet_sd, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded {len(unet_sd)} UNet parameters")

    if has_vae and not use_base_vae:
        vae_sd = {k.replace("vae.", ""): v for k, v in state_dict.items() if k.startswith("vae.")}
        pipe.vae.load_state_dict(vae_sd, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded {len(vae_sd)} VAE parameters from safetensors")
    elif use_base_vae and accelerator.is_main_process:
        print("Using base model VAE as requested")


def freeze_unet_layers(unet, strategy, layers_str, accelerator):
    """Freeze selected UNet layers to reduce trainable parameters."""
    if strategy == "none":
        return

    freeze_indices = [int(x.strip()) for x in layers_str.split(",")]
    frozen_params = 0

    for idx, block in enumerate(unet.down_blocks):
        if idx in freeze_indices:
            for param in block.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            if accelerator.is_main_process:
                print(f"Frozen down_block {idx}")

    if strategy == "color_blocks":
        unet.conv_in.requires_grad_(False)
        frozen_params += sum(p.numel() for p in unet.conv_in.parameters())
        if accelerator.is_main_process:
            print("Frozen conv_in (initial projection)")

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in unet.parameters())
        print(f"UNet: {trainable:,} trainable / {total:,} total ({trainable/total*100:.1f}%)")


def test_vae(vae, base_model, accelerator):
    """Verify VAE produces valid outputs; reload from base model if needed."""
    with torch.no_grad():
        test_image = torch.randn(1, 3, 256, 256, device=accelerator.device, dtype=vae.dtype)
        test_latent = vae.encode(test_image).latent_dist.sample()

        if torch.isnan(test_latent).any():
            print("WARNING: VAE produced NaN, reloading from base model...")
            vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float32)
            vae.eval()
            vae.requires_grad_(False)
            vae = vae.to(accelerator.device)

            test_latent = vae.encode(test_image).latent_dist.sample()
            if torch.isnan(test_latent).any():
                raise RuntimeError("VAE produces NaN outputs even from base model")
            print("Reloaded working VAE from base model")
        else:
            print(f"VAE OK: latent shape={test_latent.shape}, range=[{test_latent.min().item():.4f}, {test_latent.max().item():.4f}]")

    return vae


def load_datasets(dataset_names, split, shuffle, shuffle_seed, accelerator):
    """Load and optionally concatenate/shuffle HuggingFace datasets."""
    datasets_list = []
    for name in dataset_names:
        if accelerator.is_main_process:
            print(f"  Loading: {name}")
        ds = load_dataset(name, split=split)
        datasets_list.append(ds)
        if accelerator.is_main_process:
            print(f"    {len(ds)} samples")

    if len(datasets_list) > 1:
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(datasets_list)
        if accelerator.is_main_process:
            print(f"Concatenated: {len(dataset)} total samples")
    else:
        dataset = datasets_list[0]

    if shuffle:
        if accelerator.is_main_process:
            print(f"Shuffling with seed {shuffle_seed}")
        dataset = dataset.shuffle(seed=shuffle_seed)

    # Verify required columns
    if len(dataset) > 0:
        first = dataset[0]
        missing = [f for f in ["prompt", "chosen", "rejected"] if f not in first]
        if missing:
            raise ValueError(f"Dataset missing required fields: {missing}")

    return dataset


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    # --- Load model ---
    if accelerator.is_main_process:
        print("Loading SDXL model components...")

    dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    model_dtype = dtype_map.get(args.mixed_precision, torch.float32)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=model_dtype,
        use_safetensors=True,
        variant="fp16" if args.mixed_precision == "fp16" else None,
    )

    if accelerator.is_main_process:
        print(f"Loading weights from {args.model_path}...")
    load_model_weights(pipe, args.model_path, args.use_base_vae, accelerator)

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    # --- Memory optimizations ---
    if args.enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if accelerator.is_main_process:
            print("Gradient checkpointing enabled")

    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_xformers_memory_efficient_attention()
            if accelerator.is_main_process:
                print("xformers enabled")
        except Exception:
            if accelerator.is_main_process:
                print("xformers not available, skipping")

    vae.enable_slicing()
    vae.enable_tiling()
    vae.eval()
    vae = vae.to(dtype=torch.float32)  # float32 VAE avoids NaN
    if accelerator.is_main_process:
        print("VAE set to float32 for stability")

    # Freeze non-trainable components
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    freeze_unet_layers(unet, args.freeze_unet_strategy, args.freeze_unet_layers, accelerator)

    # --- Dataset ---
    if accelerator.is_main_process:
        print("Loading dataset(s)...")
    shuffle_seed = args.dataset_seed if args.dataset_seed is not None else args.seed
    dataset = load_datasets(args.dataset, args.dataset_split, args.shuffle_dataset, shuffle_seed, accelerator)
    if accelerator.is_main_process:
        print(f"Final dataset: {len(dataset)} samples")

    train_dataset = DPODataset(dataset, tokenizer, tokenizer_2, image_size=args.image_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Trainer ---
    trainer = DPOTrainer(
        unet=unet, vae=vae,
        text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        noise_scheduler=noise_scheduler, accelerator=accelerator,
        beta=args.beta, sft_weight=args.sft_weight, logit_clamp=args.logit_clamp,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        grad_clip_norm=args.grad_clip_norm,
        use_8bit_adam=args.use_8bit_adam, use_adafactor=args.use_adafactor,
        adafactor_scale_parameter=args.adafactor_scale_parameter,
        adafactor_relative_step=args.adafactor_relative_step,
        adafactor_warmup_init=args.adafactor_warmup_init,
        debug=args.debug,
    )

    # --- Prepare for distributed training ---
    unet, train_dataloader = accelerator.prepare(unet, train_dataloader)
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    vae.to(accelerator.device)

    # Test VAE
    if accelerator.is_main_process:
        vae = test_vae(vae, args.base_model, accelerator)
        trainer.vae = vae

    # --- W&B ---
    use_wandb = False
    if args.use_wandb and WANDB_AVAILABLE and accelerator.is_main_process:
        wandb_run_name = args.wandb_run_name or (
            f"dpo_beta{args.beta}_lr{args.learning_rate}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args),
            tags=args.wandb_tags or ["dpo", "sdxl"],
        )
        print(f"W&B run: {wandb.run.name}")
        use_wandb = True
    trainer.use_wandb = use_wandb

    # Pass scheduling config to trainer
    steps_per_epoch = len(train_dataloader)
    total_training_steps = int(args.num_epochs * steps_per_epoch)
    trainer.lr_schedule = args.lr_scheduler
    trainer.lr_warmup_steps = args.lr_warmup_steps
    trainer.lr_min = args.lr_min
    trainer.lr_cycles = args.lr_cycles
    trainer.beta_schedule = args.beta_schedule
    trainer.beta_warmup_steps = args.beta_warmup_steps
    trainer.total_steps = total_training_steps

    if accelerator.is_main_process:
        print(f"\nTraining: {args.num_epochs} epochs, {steps_per_epoch} steps/epoch, {total_training_steps} total steps")

    # --- Training loop ---
    global_step = 0
    history = {
        "epoch": [], "avg_dpo_loss": [], "avg_sft_loss": [], "avg_total_loss": [],
        "step_losses": [],
    }

    current_epoch = 0
    epoch_losses = []

    for step_idx in range(total_training_steps):
        epoch_float = step_idx / steps_per_epoch
        new_epoch = int(epoch_float)

        # Epoch boundary
        if new_epoch > current_epoch:
            if epoch_losses and accelerator.is_main_process:
                avg_dpo = np.mean([l["dpo_loss"] for l in epoch_losses])
                avg_sft = np.mean([l["sft_loss"] for l in epoch_losses])
                avg_total = np.mean([l["total_loss"] for l in epoch_losses])
                history["epoch"].append(current_epoch + 1)
                history["avg_dpo_loss"].append(avg_dpo)
                history["avg_sft_loss"].append(avg_sft)
                history["avg_total_loss"].append(avg_total)
                print(f"Epoch {current_epoch + 1} - DPO: {avg_dpo:.4f}, SFT: {avg_sft:.4f}, Total: {avg_total:.4f}")
                if use_wandb:
                    wandb.log({"epoch/avg_dpo_loss": avg_dpo, "epoch/avg_sft_loss": avg_sft,
                               "epoch/avg_total_loss": avg_total, "epoch/number": current_epoch + 1}, step=global_step)
            epoch_losses = []
            current_epoch = new_epoch

        # Get batch (cycle dataloader for fractional epochs)
        batch_idx = step_idx % steps_per_epoch
        if batch_idx == 0:
            dataloader_iter = iter(train_dataloader)
            progress_bar = tqdm(
                total=min(steps_per_epoch, total_training_steps - step_idx),
                disable=not accelerator.is_local_main_process,
                desc=f"Epoch {current_epoch + 1}",
            )

        batch = next(dataloader_iter)
        loss_dict = trainer.train_step(batch)
        trainer.global_step += 1
        epoch_losses.append(loss_dict)

        if accelerator.is_local_main_process:
            postfix = {"dpo": f"{loss_dict['dpo_loss']:.4f}", "sft": f"{loss_dict['sft_loss']:.4f}"}
            if "lr" in loss_dict:
                postfix["lr"] = f"{loss_dict['lr']:.2e}"
            progress_bar.set_postfix(postfix)
            progress_bar.update(1)

        if accelerator.is_main_process:
            history["step_losses"].append({
                "step": global_step, "epoch": epoch_float,
                "dpo_loss": loss_dict["dpo_loss"], "sft_loss": loss_dict["sft_loss"],
                "total_loss": loss_dict["total_loss"],
            })

            if use_wandb:
                log = {
                    "train/dpo_loss": loss_dict["dpo_loss"],
                    "train/sft_loss": loss_dict["sft_loss"],
                    "train/total_loss": loss_dict["total_loss"],
                    "train/epoch": epoch_float,
                }
                if "lr" in loss_dict:
                    log["train/learning_rate"] = loss_dict["lr"]
                if "grad_norm" in loss_dict:
                    log["train/grad_norm"] = loss_dict["grad_norm"]
                if trainer.beta_schedule != "constant":
                    log["train/beta"] = trainer.get_beta(global_step)
                wandb.log(log, step=global_step)

            global_step += 1

            # Save checkpoint
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                accelerator.save_state(save_path)
                unwrapped = accelerator.unwrap_model(unet)
                unwrapped.save_pretrained(os.path.join(save_path, "unet"), safe_serialization=True)
                print(f"Checkpoint saved at step {global_step}")

    # Record final epoch
    if epoch_losses and accelerator.is_main_process:
        avg_dpo = np.mean([l["dpo_loss"] for l in epoch_losses])
        avg_sft = np.mean([l["sft_loss"] for l in epoch_losses])
        avg_total = np.mean([l["total_loss"] for l in epoch_losses])
        history["epoch"].append(current_epoch + 1)
        history["avg_dpo_loss"].append(avg_dpo)
        history["avg_sft_loss"].append(avg_sft)
        history["avg_total_loss"].append(avg_total)
        print(f"Epoch {current_epoch + 1} - DPO: {avg_dpo:.4f}, SFT: {avg_sft:.4f}, Total: {avg_total:.4f}")

    # --- Save final model ---
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)

        unwrapped = accelerator.unwrap_model(unet)
        unwrapped.save_pretrained(os.path.join(final_path, "unet"), safe_serialization=True)
        pipe.unet = unwrapped
        pipe.save_pretrained(final_path, safe_serialization=True)

        # Save training history
        history_path = os.path.join(args.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        try:
            create_loss_plots(history, args.output_dir)
            print(f"Loss plots saved to {args.output_dir}/")
            if use_wandb:
                for name in ["loss_plots.png", "epoch_losses.png"]:
                    path = os.path.join(args.output_dir, name)
                    if os.path.exists(path):
                        wandb.log({name.replace(".png", ""): wandb.Image(path)})
        except Exception as e:
            print(f"Could not create plots: {e}")

        print(f"\nTraining complete! Model saved to {final_path}")

        if use_wandb:
            wandb.summary["final_dpo_loss"] = history["avg_dpo_loss"][-1]
            wandb.summary["final_sft_loss"] = history["avg_sft_loss"][-1]
            if len(history["avg_dpo_loss"]) > 1:
                wandb.summary["dpo_loss_reduction_pct"] = (
                    (history["avg_dpo_loss"][0] - history["avg_dpo_loss"][-1]) / history["avg_dpo_loss"][0] * 100
                )
            wandb.summary["total_steps"] = global_step
            wandb.finish()


if __name__ == "__main__":
    main()

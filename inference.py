#!/usr/bin/env python3
"""Run inference with a trained SDXL model (diffusers pipeline format)."""

import os
import argparse
from datetime import datetime

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Test inference with a trained SDXL model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model directory (diffusers pipeline)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="worst quality, low quality, blurry")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./test_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.float16,
            use_safetensors=True, variant="fp16",
        )
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.float16, use_safetensors=True,
        )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_vae_slicing()

    print(f"Generating {args.num_images} images...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {args.width}x{args.height}, Steps: {args.num_inference_steps}, CFG: {args.guidance_scale}")

    for i in range(args.num_images):
        generator = torch.Generator(device="cuda").manual_seed(args.seed + i)
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width, height=args.height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(args.output_dir, f"{timestamp}_seed{args.seed + i}.png")
        image.save(filepath)
        print(f"  [{i+1}/{args.num_images}] Saved {filepath}")

    print(f"Done! Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()

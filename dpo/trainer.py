import torch
import torch.nn.functional as F
import numpy as np


class DPOTrainer:
    """DPO trainer for SDXL diffusion models.

    Implements Direct Preference Optimization for diffusion models with:
    - Configurable beta and SFT regularization weight
    - LR scheduling (constant, linear, cosine, cosine_with_restarts, polynomial, exponential)
    - Beta scheduling (constant, linear, cosine)
    - Multiple optimizer backends (AdamW, 8-bit Adam, Adafactor)
    - Gradient clipping and accumulation
    """

    def __init__(
        self,
        unet,
        vae,
        text_encoder,
        text_encoder_2,
        noise_scheduler,
        accelerator,
        beta=0.1,
        sft_weight=0.1,
        logit_clamp=5.0,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_weight_decay=1e-2,
        adam_epsilon=1e-8,
        gradient_accumulation_steps=1,
        grad_clip_norm=1.0,
        use_8bit_adam=False,
        use_adafactor=False,
        adafactor_scale_parameter=False,
        adafactor_relative_step=False,
        adafactor_warmup_init=False,
        debug=False,
    ):
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.noise_scheduler = noise_scheduler
        self.accelerator = accelerator
        self.beta = beta
        self.sft_weight = sft_weight
        self.logit_clamp = logit_clamp
        self.debug = debug
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip_norm = grad_clip_norm

        # LR scheduling state
        self.lr_schedule = "constant"
        self.base_lr = learning_rate
        self.lr_warmup_steps = 100
        self.lr_min = 1e-9
        self.lr_cycles = 1
        self.current_step = 0
        self.total_steps = 1
        self.global_step = 0

        # Beta scheduling state
        self.beta_schedule = "constant"
        self.beta_warmup_steps = 100

        # W&B state
        self.use_wandb = False

        # Setup optimizer
        self.optimizer = self._create_optimizer(
            learning_rate, adam_beta1, adam_beta2, adam_weight_decay, adam_epsilon,
            use_8bit_adam, use_adafactor,
            adafactor_scale_parameter, adafactor_relative_step, adafactor_warmup_init,
        )

    def _create_optimizer(
        self, lr, beta1, beta2, weight_decay, epsilon,
        use_8bit_adam, use_adafactor,
        scale_parameter, relative_step, warmup_init,
    ):
        if use_adafactor:
            try:
                from transformers import Adafactor
                optimizer = Adafactor(
                    self.unet.parameters(),
                    lr=lr if not relative_step else None,
                    scale_parameter=scale_parameter,
                    relative_step=relative_step,
                    warmup_init=warmup_init,
                    weight_decay=weight_decay,
                    eps=(epsilon, 1e-3),
                    clip_threshold=1.0,
                    beta1=None,
                )
                print(f"Using Adafactor optimizer (scale_parameter={scale_parameter}, relative_step={relative_step})")
                return optimizer
            except ImportError:
                print("Adafactor not available, falling back to AdamW")

        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
                print("Using 8-bit Adam optimizer")
                return bnb.optim.AdamW8bit(
                    self.unet.parameters(), lr=lr,
                    betas=(beta1, beta2), weight_decay=weight_decay, eps=epsilon,
                )
            except ImportError:
                print("bitsandbytes not available, using standard AdamW")

        return torch.optim.AdamW(
            self.unet.parameters(), lr=lr,
            betas=(beta1, beta2), weight_decay=weight_decay, eps=epsilon,
        )

    def get_lr(self, step):
        """Compute learning rate with warmup and scheduling."""
        if self.lr_schedule == "constant":
            return self.base_lr

        # Warmup phase
        if step < self.lr_warmup_steps:
            return self.lr_min + (self.base_lr - self.lr_min) * (step / self.lr_warmup_steps)

        progress = (step - self.lr_warmup_steps) / max(1, self.total_steps - self.lr_warmup_steps)
        progress = min(1.0, progress)

        if self.lr_schedule == "linear":
            return self.base_lr + (self.lr_min - self.base_lr) * progress
        elif self.lr_schedule == "cosine":
            return self.lr_min + (self.base_lr - self.lr_min) * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.lr_schedule == "cosine_with_restarts":
            cycle_progress = (progress * self.lr_cycles) % 1.0
            return self.lr_min + (self.base_lr - self.lr_min) * 0.5 * (1 + np.cos(np.pi * cycle_progress))
        elif self.lr_schedule == "polynomial":
            return self.base_lr * (1 - progress) ** 0.9
        elif self.lr_schedule == "exponential":
            return self.base_lr * (self.lr_min / self.base_lr) ** progress

        return self.base_lr

    def get_beta(self, step):
        """Compute beta value with optional scheduling."""
        if self.beta_schedule == "linear":
            if step < self.beta_warmup_steps:
                return self.beta * (step / self.beta_warmup_steps)
            tail_start = int(0.7 * self.total_steps)
            if step >= tail_start:
                frac = (step - tail_start) / max(1, self.total_steps - tail_start)
                return self.beta + (0.3 - self.beta) * frac
            return self.beta
        elif self.beta_schedule == "cosine":
            progress = step / self.total_steps
            return self.beta * 0.5 * (1 + np.cos(np.pi * (progress - 1)))
        return self.beta

    def encode_prompt(self, batch):
        """Encode text prompts using both SDXL text encoders."""
        device = self.text_encoder.device
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        input_ids2 = batch["input_ids_2"].to(device)
        attn2 = batch["attention_mask_2"].to(device)

        out1 = self.text_encoder(input_ids, attention_mask=attn, output_hidden_states=True)
        prompt_embeds_1 = out1.hidden_states[-2]

        out2 = self.text_encoder_2(input_ids2, attention_mask=attn2, output_hidden_states=True)
        pooled_prompt_embeds = out2.text_embeds
        prompt_embeds_2 = out2.hidden_states[-2]

        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        bsz = input_ids.shape[0]
        time_ids = torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]] * bsz,
            dtype=torch.float32, device=device,
        )

        return prompt_embeds, pooled_prompt_embeds, time_ids

    def compute_loss(self, batch):
        """Compute DPO + SFT loss for a batch."""
        prompt_embeds, pooled_prompt_embeds, time_ids = self.encode_prompt(batch)

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": time_ids,
        }

        chosen_image = batch["chosen_image"].to(dtype=torch.float32, device=self.vae.device)
        rejected_image = batch["rejected_image"].to(dtype=torch.float32, device=self.vae.device)

        if self.debug and not hasattr(self, "_input_debug_counter"):
            self._input_debug_counter = 0
        if self.debug and self._input_debug_counter < 3:
            print(f"\nDebug input step {self._input_debug_counter}:")
            print(f"  chosen_image range: [{chosen_image.min().item():.4f}, {chosen_image.max().item():.4f}]")
            print(f"  rejected_image range: [{rejected_image.min().item():.4f}, {rejected_image.max().item():.4f}]")
            self._input_debug_counter += 1

        # Encode images to latent space
        with torch.no_grad():
            chosen_latents = self.vae.encode(chosen_image).latent_dist.sample()
            rejected_latents = self.vae.encode(rejected_image).latent_dist.sample()

            if torch.isnan(chosen_latents).any() or torch.isnan(rejected_latents).any():
                print("ERROR: VAE produced NaN latents!")
                return torch.tensor(0.0, device=chosen_image.device, requires_grad=True), {
                    "dpo_loss": 0.0, "sft_loss": 0.0, "total_loss": 0.0,
                }

            chosen_latents = chosen_latents * self.vae.config.scaling_factor
            rejected_latents = rejected_latents * self.vae.config.scaling_factor

        # Sample noise and timesteps (biased toward mid-range for stability)
        noise = torch.randn_like(chosen_latents)
        T = self.noise_scheduler.config.num_train_timesteps
        u = torch.rand(chosen_latents.shape[0], device=chosen_latents.device)
        u = 0.3 + 0.5 * u  # sample in [30%, 80%]
        timesteps = (u * T).long().clamp_(0, T - 1)

        noisy_chosen = self.noise_scheduler.add_noise(chosen_latents, noise, timesteps)
        noisy_rejected = self.noise_scheduler.add_noise(rejected_latents, noise, timesteps)

        # Predict noise for both chosen and rejected
        pred_noise_chosen = self.unet(
            noisy_chosen, timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        pred_noise_rejected = self.unet(
            noisy_rejected, timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # Compute losses in float32 for numerical stability
        pred_noise_chosen = pred_noise_chosen.float()
        pred_noise_rejected = pred_noise_rejected.float()
        noise = noise.float()

        loss_chosen = F.mse_loss(pred_noise_chosen, noise, reduction="mean")
        loss_rejected = F.mse_loss(pred_noise_rejected, noise, reduction="mean")

        # Per-sample losses for DPO
        loss_chosen_per_sample = F.mse_loss(pred_noise_chosen, noise, reduction="none").mean(dim=[1, 2, 3])
        loss_rejected_per_sample = F.mse_loss(pred_noise_rejected, noise, reduction="none").mean(dim=[1, 2, 3])

        # DPO log-ratio with clamping for stability
        pi_logratios = loss_rejected_per_sample - loss_chosen_per_sample
        pi_logratios = torch.clamp(pi_logratios, min=-self.logit_clamp, max=self.logit_clamp)

        current_beta = self.get_beta(self.global_step)
        dpo_loss = -F.logsigmoid(current_beta * pi_logratios).mean()

        # SFT regularization on chosen examples
        sft_loss = loss_chosen

        total_loss = dpo_loss + self.sft_weight * sft_loss

        return total_loss, {
            "dpo_loss": dpo_loss.item() if not torch.isnan(dpo_loss) else 0.0,
            "sft_loss": sft_loss.item() if not torch.isnan(sft_loss) else 0.0,
            "total_loss": total_loss.item() if not torch.isnan(total_loss) else 0.0,
        }

    def train_step(self, batch):
        """Execute one training step with gradient accumulation."""
        # Update learning rate
        if self.lr_schedule != "constant":
            current_lr = self.get_lr(self.current_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

        with self.accelerator.accumulate(self.unet):
            loss, loss_dict = self.compute_loss(batch)
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.unet.parameters(), self.grad_clip_norm,
                )
                loss_dict["grad_norm"] = grad_norm.item()

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.current_step += 1

        if self.lr_schedule != "constant":
            loss_dict["lr"] = current_lr

        return loss_dict

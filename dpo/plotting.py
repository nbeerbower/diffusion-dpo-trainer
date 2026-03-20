import os
import numpy as np
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def create_loss_plots(history, output_dir):
    """Create and save training loss plots."""
    if not PLOTTING_AVAILABLE:
        print("matplotlib not available, skipping plot generation")
        return

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Epoch-averaged losses
    ax1 = axes[0, 0]
    epochs = history["epoch"]
    ax1.plot(epochs, history["avg_dpo_loss"], "b-", label="DPO Loss", linewidth=2)
    ax1.plot(epochs, history["avg_sft_loss"], "g-", label="SFT Loss", linewidth=2)
    ax1.plot(epochs, history["avg_total_loss"], "r-", label="Total Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Average Losses per Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Step-wise DPO loss with moving average
    ax2 = axes[0, 1]
    steps = [item["step"] for item in history["step_losses"]]
    dpo_losses = [item["dpo_loss"] for item in history["step_losses"]]
    ax2.plot(steps, dpo_losses, "b-", alpha=0.6, linewidth=1)
    window = min(50, len(dpo_losses) // 4)
    if window > 1:
        dpo_ma = np.convolve(dpo_losses, np.ones(window) / window, mode="valid")
        ax2.plot(steps[window - 1:], dpo_ma, "b-", linewidth=2, label=f"MA{window}")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("DPO Loss")
    ax2.set_title("DPO Loss over Training Steps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Step-wise SFT loss with moving average
    ax3 = axes[1, 0]
    sft_losses = [item["sft_loss"] for item in history["step_losses"]]
    ax3.plot(steps, sft_losses, "g-", alpha=0.6, linewidth=1)
    if window > 1:
        sft_ma = np.convolve(sft_losses, np.ones(window) / window, mode="valid")
        ax3.plot(steps[window - 1:], sft_ma, "g-", linewidth=2, label=f"MA{window}")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("SFT Loss")
    ax3.set_title("SFT Loss over Training Steps")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Training statistics summary
    ax4 = axes[1, 1]
    ax4.axis("off")
    final_dpo = history["avg_dpo_loss"][-1]
    initial_dpo = history["avg_dpo_loss"][0]
    dpo_reduction = (initial_dpo - final_dpo) / initial_dpo * 100
    final_sft = history["avg_sft_loss"][-1]
    initial_sft = history["avg_sft_loss"][0]

    stats_text = (
        f"Training Statistics:\n\n"
        f"Initial DPO Loss: {initial_dpo:.4f}\n"
        f"Final DPO Loss: {final_dpo:.4f}\n"
        f"DPO Loss Reduction: {dpo_reduction:.1f}%\n\n"
        f"Initial SFT Loss: {initial_sft:.4f}\n"
        f"Final SFT Loss: {final_sft:.4f}\n\n"
        f"Total Steps: {len(history['step_losses'])}\n"
        f"Total Epochs: {len(history['epoch'])}\n\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    ax4.text(
        0.1, 0.5, stats_text,
        transform=ax4.transAxes, fontsize=12,
        verticalalignment="center", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Simple epoch-level plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["avg_dpo_loss"], "b-o", label="DPO Loss", linewidth=2, markersize=8)
    plt.plot(epochs, history["avg_sft_loss"], "g-o", label="SFT Loss", linewidth=2, markersize=8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "epoch_losses.png"), dpi=150, bbox_inches="tight")
    plt.close()

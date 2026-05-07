"""Train the diffusion model for a given theme.

Run: python diffusion_train_model.py --theme City
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.diffusion_dataset import DiffusionDataset
from src.model.diffusion_config import DiffusionConfig
from src.model.diffusion_train import DiffusionTrainer
from src.utils.logging import setup_logging

logger = setup_logging()


def _plot_losses(history: dict, out_path: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], label="total")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for key in ("pos", "part", "color", "rot"):
        axes[1].plot(epochs, history[key], label=key)
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Loss components")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Loss plot saved to %s", out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", required=True)
    parser.add_argument("--data-dir", default="dataset/diffusion_sets")
    parser.add_argument("--ckpt-dir", default="checkpoints/diffusion")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-sets", type=int, default=None, help="Limit dataset to first N sets (overfit test)")
    args = parser.parse_args()

    vocab_path = Path(args.data_dir) / f"vocab_{args.theme}.pt"
    if not vocab_path.exists():
        logger.error("Vocab not found: %s — run diffusion_build_dataset.py first", vocab_path)
        return

    vocab = torch.load(vocab_path, weights_only=True)

    cfg = DiffusionConfig(
        theme=args.theme,
        max_bricks=vocab["max_bricks"],
        n_parts=vocab["n_parts"],
        n_colors=vocab["n_colors"],
        batch_size=args.batch_size,
        max_epochs=args.epochs,
    )

    dataset = DiffusionDataset(Path(args.data_dir) / args.theme, max_sets=args.max_sets)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(args.device == "cuda"),
    )
    logger.info(
        "Dataset: %d sets, N=%d, %d parts, %d colors",
        len(dataset), cfg.max_bricks, cfg.n_parts, cfg.n_colors,
    )

    ckpt_dir = Path(args.ckpt_dir) / args.theme
    trainer = DiffusionTrainer(cfg, args.device)

    if not args.no_resume:
        trainer.load_checkpoint(ckpt_dir)

    history: dict[str, list[float]] = {"loss": [], "pos": [], "part": [], "color": [], "rot": []}

    for epoch in range(cfg.max_epochs):
        totals: dict[str, float] = {"loss": 0.0, "pos": 0.0, "part": 0.0, "color": 0.0, "rot": 0.0}
        for batch in loader:
            m = trainer.train_step(batch)
            for k in totals:
                totals[k] += m[k] if k != "loss" else m["loss"]

        trainer.scheduler.step()

        n = len(loader)
        for k in totals:
            history[k].append(totals[k] / n)

        logger.info(
            "Epoch %d/%d | loss=%.4f | pos=%.4f part=%.4f color=%.4f rot=%.4f",
            epoch + 1, cfg.max_epochs,
            history["loss"][-1], history["pos"][-1],
            history["part"][-1], history["color"][-1], history["rot"][-1],
        )

        if trainer.global_step % cfg.checkpoint_interval == 0:
            trainer.save_checkpoint(ckpt_dir)

    trainer.save_checkpoint(ckpt_dir)
    _plot_losses(history, ckpt_dir / "loss_curves.png")
    logger.info("Training complete")


if __name__ == "__main__":
    main()

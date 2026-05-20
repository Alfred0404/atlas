"""Train the diffusion model for a given theme.

Run: python diffusion_train_model.py --theme City
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, random_split

from src.data.diffusion_dataset import DiffusionDataset
from src.model.diffusion_config import DiffusionConfig
from src.model.diffusion_train import DiffusionTrainer
from src.utils.logging import setup_logging

logger = setup_logging()


def _plot_losses(history: list[float], out_path: Path, val_history: list[float] = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history) + 1)
    ax.plot(epochs, history, label="train")
    if val_history:
        ax.plot(epochs, val_history, label="val", linestyle="--")
    ax.set_ylabel("MSE noise loss")
    ax.set_xlabel("Epoch")
    ax.set_title("Position-denoising loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
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

    full_dataset = DiffusionDataset(Path(args.data_dir) / args.theme, max_sets=args.max_sets)

    pin = args.device == "cuda"
    val_loader = None
    if len(full_dataset) >= 5:
        val_size = max(1, len(full_dataset) // 10)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=2, pin_memory=pin)
    else:
        train_dataset = full_dataset

    if len(train_dataset) < cfg.batch_size:
        sampler = RandomSampler(train_dataset, replacement=True, num_samples=cfg.batch_size)
        loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler,
                            num_workers=0, pin_memory=pin)
    else:
        loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=2, pin_memory=pin)
    logger.info(
        "Dataset: %d train / %d val sets, N=%d, %d parts, %d colors",
        len(train_dataset), len(val_dataset) if val_loader else 0,
        cfg.max_bricks, cfg.n_parts, cfg.n_colors,
    )

    ckpt_dir = Path(args.ckpt_dir) / args.theme
    trainer = DiffusionTrainer(cfg, args.device)

    if not args.no_resume:
        trainer.load_checkpoint(ckpt_dir)

    history: list[float] = []
    val_history: list[float] = [] if val_loader else None

    for epoch in range(cfg.max_epochs):
        total = 0.0
        for batch in loader:
            total += trainer.train_step(batch)["loss"]

        trainer.scheduler.step()
        history.append(total / len(loader))

        val_suffix = ""
        if val_loader:
            v = sum(trainer.eval_step(b)["loss"] for b in val_loader) / len(val_loader)
            val_history.append(v)
            val_suffix = f" | val_loss={v:.4f}"

        logger.info("Epoch %d/%d | loss=%.4f%s",
                    epoch + 1, cfg.max_epochs, history[-1], val_suffix)

        if trainer.global_step > 0 and trainer.global_step % cfg.checkpoint_interval == 0:
            trainer.save_checkpoint(ckpt_dir)

    trainer.save_checkpoint(ckpt_dir)
    _plot_losses(history, ckpt_dir / "loss_curves.png", val_history)
    logger.info("Training complete")


if __name__ == "__main__":
    main()

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


def _plot_losses(history: dict, out_path: Path, val_history: dict = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], label="train")
    if val_history:
        axes[0].plot(epochs, val_history["loss"], label="val", linestyle="--")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for key in ("pos", "part", "color", "rot"):
        l, = axes[1].plot(epochs, history[key], label=key)
        if val_history:
            axes[1].plot(epochs, val_history[key], linestyle="--", color=l.get_color())
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Loss components (solid=train, dashed=val)")
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
    parser.add_argument("--discrete-t-max", type=int, default=500, help="Only compute discrete CE loss when t < this value (default=T, i.e. all steps)")
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
        discrete_t_max=args.discrete_t_max,
    )

    full_dataset = DiffusionDataset(Path(args.data_dir) / args.theme, max_sets=args.max_sets)

    # Train / val split (90/10). Disabled for tiny datasets (< 5 sets) to avoid empty val.
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

    # When dataset is smaller than batch_size (e.g. single-set overfit), sample with
    # replacement to fill a full batch — each slot gets a different t in _loss,
    # giving a stable gradient across timesteps rather than one noisy t per epoch.
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

    _keys = ("loss", "pos", "part", "color", "rot")
    history: dict[str, list[float]] = {k: [] for k in _keys}
    val_history: dict[str, list[float]] = {k: [] for k in _keys} if val_loader else None

    for epoch in range(cfg.max_epochs):
        totals: dict[str, float] = {k: 0.0 for k in _keys}
        for batch in loader:
            m = trainer.train_step(batch)
            for k in _keys:
                totals[k] += m[k]

        trainer.scheduler.step()

        n = len(loader)
        for k in _keys:
            history[k].append(totals[k] / n)

        val_suffix = ""
        if val_loader:
            val_totals: dict[str, float] = {k: 0.0 for k in _keys}
            for batch in val_loader:
                m = trainer.eval_step(batch)
                for k in _keys:
                    val_totals[k] += m[k]
            nv = len(val_loader)
            for k in _keys:
                val_history[k].append(val_totals[k] / nv)
            val_suffix = f" | val_loss={val_history['loss'][-1]:.4f}"

        logger.info(
            "Epoch %d/%d | loss=%.4f | pos=%.4f part=%.4f color=%.4f rot=%.4f%s",
            epoch + 1, cfg.max_epochs,
            history["loss"][-1], history["pos"][-1],
            history["part"][-1], history["color"][-1], history["rot"][-1],
            val_suffix,
        )

        if trainer.global_step > 0 and trainer.global_step % cfg.checkpoint_interval == 0:
            trainer.save_checkpoint(ckpt_dir)

    trainer.save_checkpoint(ckpt_dir)
    _plot_losses(history, ckpt_dir / "loss_curves.png", val_history)
    logger.info("Training complete")


if __name__ == "__main__":
    main()

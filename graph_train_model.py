"""Entry point for Graph Transformer training.

Usage
-----
    python graph_train_model.py --theme "City" [--device cuda|cpu]

Expects:
  - dataset/graph_sets/<theme>/*.npz  — produced by graph_build_dataset.py --theme
  - dataset/graph_vocab_<theme>.pt    — vocabulary saved by graph_build_dataset.py --theme
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.model.graph_config import GraphModelConfig
from src.model.graph_train import build_trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Graph Transformer model.")
    parser.add_argument("--theme", required=True, help="Theme name (e.g. 'City'). Determines default graph-dir, vocab-path, and checkpoint-dir.")
    parser.add_argument(
        "--graph-dir",
        default=None,
        help="Override graph dataset directory",
    )
    parser.add_argument(
        "--vocab-path",
        default=None,
        help="Override vocabulary file path",
    )
    parser.add_argument("--device", default=None, help="Device (default: auto-detect)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Ignore existing checkpoint and train from scratch",
    )
    parser.add_argument(
        "--live-plot",
        action="store_true",
        default=False,
        help="Show a live loss plot (requires matplotlib)",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    theme = args.theme
    graph_dir = args.graph_dir or f"dataset/graph_sets/{theme}"
    vocab_path = args.vocab_path or f"dataset/graph_vocab_{theme}.pt"
    checkpoint_dir = f"./checkpoints/graph/{theme}"

    logger.info("Starting graph training entrypoint")
    logger.info("theme=%s  graph_dir=%s", theme, graph_dir)

    # Load vocabulary to set correct n_parts / n_colors
    cfg = GraphModelConfig()
    cfg.checkpoint_dir = checkpoint_dir
    vpath = Path(vocab_path)
    if vpath.exists():
        saved = torch.load(vpath, weights_only=True)
        cfg.n_parts = len(saved["part_vocab"]) + 1  # +1 for index-0 unknown
        cfg.n_colors = len(saved["color_vocab"]) + 1
        logging.getLogger(__name__).info(
            "Vocabulary loaded — %d parts, %d colors", cfg.n_parts, cfg.n_colors
        )
    else:
        logging.getLogger(__name__).warning(
            "Vocabulary file not found at %s — using defaults (%d parts, %d colors)",
            vpath,
            cfg.n_parts,
            cfg.n_colors,
        )

    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    selected_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Config: epochs=%d batch_size=%d device=%s",
        cfg.max_epochs,
        cfg.batch_size,
        selected_device,
    )

    trainer = build_trainer(
        graph_dir=graph_dir,
        cfg=cfg,
        live_plot=args.live_plot,
        **({"device": args.device} if args.device else {}),
    )

    latest = Path(cfg.checkpoint_dir) / "latest.pt"
    if not args.no_resume and latest.exists():
        logger.info("Resuming from checkpoint: %s", latest)
        trainer.load_checkpoint("latest.pt")
    else:
        logger.info("No checkpoint found — training from scratch")

    trainer.train()


if __name__ == "__main__":
    main()

"""Entry point for Graph Transformer training.

Usage
-----
    python graph_train_model.py [--graph-dir PATH] [--device cuda|cpu]

Expects:
  - dataset/graph_sets/*.npz  — produced by graph_build_dataset.py
  - dataset/graph_vocab.pt    — vocabulary saved by graph_build_dataset.py
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

VOCAB_PATH = Path("dataset/graph_vocab.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Graph Transformer model.")
    parser.add_argument(
        "--graph-dir",
        default="dataset/graph_sets",
        help="Directory containing .npz graph files (default: dataset/graph_sets)",
    )
    parser.add_argument(
        "--vocab-path",
        default=str(VOCAB_PATH),
        help="Vocabulary file saved by graph_build_dataset.py",
    )
    parser.add_argument("--device", default=None, help="Device (default: auto-detect)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("Starting graph training entrypoint")
    logger.info("graph_dir=%s", args.graph_dir)

    # Load vocabulary to set correct n_parts / n_colors
    cfg = GraphModelConfig()
    vpath = Path(args.vocab_path)
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
        graph_dir=args.graph_dir,
        cfg=cfg,
        **({"device": args.device} if args.device else {}),
    )
    trainer.train()


if __name__ == "__main__":
    main()

"""Entry point for Graph Transformer generation.

Usage
-----
    python graph_generate_model.py [--checkpoint PATH] [--output PATH]
                                   [--seed-part INT] [--seed-color INT]

Requires a trained checkpoint from graph_train_model.py and a PartDatabase
pointing to your LDraw library.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.geometry.lego_part import PartDatabase
from src.model.graph_config import GraphModelConfig
from src.model.graph_generate import GraphGenerator
from src.model.graph_transformer import GraphTransformer
from src.file_io.mpd_writer import write_mpd_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a LEGO assembly.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/graph/best.pt",
        help="Path to trained checkpoint (default: checkpoints/graph/best.pt)",
    )
    parser.add_argument(
        "--output",
        default="generated_sets/graph_output.mpd",
        help="Output MPD file path",
    )
    parser.add_argument(
        "--seed-part",
        type=int,
        default=1,
        help="Vocabulary part index for seed brick (default: 1)",
    )
    parser.add_argument(
        "--seed-color",
        type=int,
        default=1,
        help="Vocabulary color index for seed brick (default: 1)",
    )
    parser.add_argument(
        "--max-bricks",
        type=int,
        default=None,
        help="Override cfg.max_gen_bricks",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (default: auto-detect cuda/cpu)",
    )
    parser.add_argument(
        "--vocab-path",
        default="dataset/graph_vocab.pt",
        help="Path to graph vocabulary saved by graph_build_dataset.py",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return

    torch.serialization.add_safe_globals([GraphModelConfig])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg: GraphModelConfig = ckpt["cfg"]

    model = GraphTransformer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Loaded checkpoint from %s (step %d)", ckpt_path, ckpt["global_step"])

    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        logger.error("Vocabulary file not found: %s", vocab_path)
        logger.error("Run graph_build_dataset.py first or pass --vocab-path")
        return

    saved_vocab = torch.load(vocab_path, map_location="cpu", weights_only=True)
    saved_part_vocab: dict[str, int] = saved_vocab["part_vocab"]
    saved_color_vocab: dict[int, int] = saved_vocab["color_vocab"]

    # Invert saved mappings to what GraphGenerator expects:
    #   part_vocab: idx -> part_id_str
    #   color_vocab: idx -> ldraw_color_int
    part_vocab = {idx: part_id for part_id, idx in saved_part_vocab.items()}
    color_vocab = {idx: color for color, idx in saved_color_vocab.items()}

    # Ensure index 0 has safe defaults (unknown token).
    part_vocab.setdefault(0, "3001")
    color_vocab.setdefault(0, 7)

    part_db = PartDatabase()

    generator = GraphGenerator(
        cfg=cfg,
        model=model,
        part_db=part_db,
        part_vocab=part_vocab,
        color_vocab=color_vocab,
        device=device,
    )

    seed_part = args.seed_part
    seed_color = args.seed_color

    # Validate seed part against loaded vocab and geometry.
    seed_part_key = part_vocab.get(seed_part)
    if seed_part_key is None or not part_db.get_or_default(seed_part_key).ports:
        logger.warning(
            "Seed part idx %d is invalid or has no ports. Selecting a fallback seed with geometry.",
            seed_part,
        )
        fallback_seed = None
        for idx in sorted(k for k in part_vocab.keys() if k != 0):
            part_key = part_vocab[idx]
            if part_db.get_or_default(part_key).ports:
                fallback_seed = idx
                break
        if (
            fallback_seed is None
            and 0 in part_vocab
            and part_db.get_or_default(part_vocab[0]).ports
        ):
            fallback_seed = 0
        if fallback_seed is None:
            logger.error("No valid seed part with geometry found in vocabulary.")
            return
        seed_part = fallback_seed
        logger.info(
            "Using fallback seed part idx %d (%s)", seed_part, part_vocab[seed_part]
        )

    if seed_color not in color_vocab:
        logger.warning(
            "Seed color idx %d not in vocabulary. Falling back to 1.", seed_color
        )
        seed_color = 1 if 1 in color_vocab else 0

    raw_bricks = generator.generate(
        seed_part_id=seed_part,
        seed_color=seed_color,
        max_bricks=args.max_bricks,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mpd_file(str(out_path), raw_bricks, model_name="graph_generated")
    logger.info("Wrote %d bricks to %s", len(raw_bricks), out_path)


if __name__ == "__main__":
    main()

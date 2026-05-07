"""Generate LEGO sets from a trained diffusion model.

Run: python diffusion_generate_model.py --theme City --checkpoint checkpoints/diffusion/City/latest.pt
"""

import argparse
from pathlib import Path

import torch

from src.model.diffusion_config import DiffusionConfig
from src.model.dit import DiT
from src.model.ddpm import DDPM
from src.model.diffusion_generate import generate
from src.file_io.mpd_writer import write_mpd_file
from src.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab-path", default=None)
    parser.add_argument("--out-dir", default="generated_sets")
    parser.add_argument("--n-sets", type=int, default=1)
    parser.add_argument("--n-bricks", type=int, default=None, help="Force exactly N bricks (takes top-N by confidence)")
    parser.add_argument("--min-confidence", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    cfg: DiffusionConfig = ckpt["cfg"]

    model = DiT(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    logger.info("Loaded checkpoint (step %d)", ckpt["global_step"])

    ddpm = DDPM(cfg.T, cfg.beta_schedule, args.device)

    vocab_path = args.vocab_path or f"dataset/diffusion_sets/vocab_{args.theme}.pt"
    vocab = torch.load(vocab_path, weights_only=True)

    logger.info("Generating %d set(s) for theme '%s'...", args.n_sets, args.theme)
    sets = generate(model, ddpm, cfg, vocab, args.device, args.n_sets, n_bricks=args.n_bricks, min_confidence=args.min_confidence)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, bricks in enumerate(sets):
        out_path = out_dir / f"generated_{args.theme}_{i}.mpd"
        write_mpd_file(str(out_path), bricks)
        logger.info("Saved %d bricks to %s", len(bricks), out_path)


if __name__ == "__main__":
    main()

"""Generate LEGO sets from a trained diffusion model.

The model only denoises positions; the bag of bricks (part/color/rotation ids)
must be supplied as conditioning. By default a random training set from the theme
is used; pass --template <set.pt> to override.

Run: python diffusion_generate_model.py --theme City --checkpoint checkpoints/diffusion/City/latest.pt
"""

import argparse
import random
from pathlib import Path

import torch

from src.model.diffusion_config import DiffusionConfig
from src.model.dit import DiT
from src.model.ddpm import DDPM
from src.model.diffusion_generate import generate
from src.file_io.mpd_writer import write_mpd_file
from src.utils.logging import setup_logging

logger = setup_logging()


def _load_bag(path: Path) -> dict:
    t = torch.load(path, weights_only=True)
    return {
        "part_ids": t["part_ids"],
        "color_ids": t["color_ids"],
        "rot_ids": t["rot_ids"],
        "padding_mask": t["padding_mask"],
        "n_bricks": int(t["n_bricks"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab-path", default=None)
    parser.add_argument("--template", default=None,
                        help="Path to a .pt set file used as the bag. If omitted, a random set from the theme is used.")
    parser.add_argument("--data-dir", default="dataset/diffusion_sets",
                        help="Used to pick a random bag when --template is not given")
    parser.add_argument("--out-dir", default="generated_sets")
    parser.add_argument("--n-sets", type=int, default=1)
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

    vocab_path = args.vocab_path or f"{args.data_dir}/vocab_{args.theme}.pt"
    vocab = torch.load(vocab_path, weights_only=True)

    if args.template:
        bag_path = Path(args.template)
    else:
        candidates = list((Path(args.data_dir) / args.theme).glob("*.pt"))
        if not candidates:
            logger.error("No training sets found in %s", Path(args.data_dir) / args.theme)
            return
        bag_path = random.choice(candidates)
    logger.info("Using bag from %s", bag_path)
    bag = _load_bag(bag_path)
    logger.info("Bag has %d bricks", bag["n_bricks"])

    logger.info("Generating %d set(s) for theme '%s'...", args.n_sets, args.theme)
    sets = generate(model, ddpm, cfg, vocab, args.device, bag, args.n_sets)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, bricks in enumerate(sets):
        out_path = out_dir / f"generated_{args.theme}_{i}.mpd"
        write_mpd_file(str(out_path), bricks)
        logger.info("Saved %d bricks to %s", len(bricks), out_path)


if __name__ == "__main__":
    main()

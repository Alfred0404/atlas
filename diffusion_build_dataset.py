"""Build diffusion dataset for a given theme.

Parses MPD files → per-set (N, 6) tensors + vocabulary.
Run: python diffusion_build_dataset.py --theme City
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.data.parser import MPDParser, RawBrickData
from src.maths.rotations import generate_chiral_rotation_matrices, find_closest_rotation_matrix
from src.maths.transforms import get_rotation_matrix_from_world_matrix
from src.utils.logging import setup_logging

logger = setup_logging()

_ROTATION_MATRICES = generate_chiral_rotation_matrices()

# --- Geometric augmentation ---
# 8 transforms: 4 Y-axis rotations × 2 mirror states (identity + mirror X)
def _y_rot(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

_MIRROR_X = np.diag([-1.0, 1.0, 1.0])
_AUG_TRANSFORMS: list[np.ndarray] = (
    [_y_rot(d) for d in (0, 90, 180, 270)]
    + [_MIRROR_X @ _y_rot(d) for d in (0, 90, 180, 270)]
)
# Precompute rot_id remap table: _ROT_REMAP[aug_idx, orig_id] = new_id
# Shape (8, 24) — computed once at import time, O(1) per brick at save time.
_ROT_REMAP: np.ndarray = np.array(
    [
        [
            find_closest_rotation_matrix(aug @ rot, _ROTATION_MATRICES)
            for rot in _ROTATION_MATRICES
        ]
        for aug in _AUG_TRANSFORMS
    ],
    dtype=np.int64,
)


def _get_model_name(mpd_path: Path) -> str:
    with open(mpd_path, "r") as f:
        for line in f:
            if line.startswith("0 FILE"):
                return line.split(maxsplit=2)[2]
    return mpd_path.stem + ".ldr\n"


def _parse_set(mpd_path: Path) -> list[RawBrickData]:
    parser = MPDParser(str(mpd_path))
    parser.parse(_get_model_name(mpd_path))
    parser.sort_bricks_by_position()
    return parser.raw_data


def _build_vocab(
    all_bricks: list[list[RawBrickData]], top_k_parts: int = None
) -> tuple[dict, dict]:
    part_counts: Counter = Counter()
    color_counts: Counter = Counter()
    for bricks in all_bricks:
        for b in bricks:
            part_counts[b.brick_id] += 1
            color_counts[b.color] += 1
    # Index 0 reserved for UNK/PAD; real entries start at 1
    top_parts = part_counts.most_common(top_k_parts)
    part_vocab = {p: i + 1 for i, (p, _) in enumerate(top_parts)}
    color_vocab = {c: i + 1 for i, (c, _) in enumerate(color_counts.most_common())}
    return part_vocab, color_vocab


def _bricks_to_raw(
    bricks: list[RawBrickData], part_vocab: dict, color_vocab: dict
) -> dict:
    N = len(bricks)
    positions = np.zeros((N, 3), dtype=np.float32)
    rot_ids = np.zeros(N, dtype=np.int64)
    part_ids = np.zeros(N, dtype=np.int64)
    color_ids = np.zeros(N, dtype=np.int64)

    for i, b in enumerate(bricks):
        positions[i] = b.world_matrix[:3, 3]
        rot = get_rotation_matrix_from_world_matrix(b.world_matrix)
        rot_ids[i] = find_closest_rotation_matrix(rot, _ROTATION_MATRICES)
        part_ids[i] = part_vocab.get(b.brick_id, 0)
        color_ids[i] = color_vocab.get(b.color, 0)

    positions -= positions.mean(axis=0)

    return {
        "positions": torch.from_numpy(positions),
        "rot_ids": torch.from_numpy(rot_ids),
        "part_ids": torch.from_numpy(part_ids),
        "color_ids": torch.from_numpy(color_ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", required=True)
    parser.add_argument("--mpd-dir", default="dataset/mpd_files")
    parser.add_argument("--out-dir", default="dataset/diffusion_sets")
    parser.add_argument("--top-k-parts", type=int, default=None, help="Keep only the K most frequent parts (others → UNK index 0)")
    args = parser.parse_args()

    theme_dir = Path(args.mpd_dir) / args.theme
    out_dir = Path(args.out_dir) / args.theme
    out_dir.mkdir(parents=True, exist_ok=True)

    mpd_files = list(theme_dir.glob("*.mpd"))
    if not mpd_files:
        logger.error("No MPD files in %s", theme_dir)
        return

    logger.info("Found %d MPD files for theme '%s'", len(mpd_files), args.theme)

    # Pass 1: parse all sets and build vocabulary
    all_bricks: list[list[RawBrickData]] = []
    valid_files: list[Path] = []
    for mpd_path in mpd_files:
        try:
            bricks = _parse_set(mpd_path)
            if bricks:
                all_bricks.append(bricks)
                valid_files.append(mpd_path)
        except Exception as e:
            logger.warning("Skipping %s: %s", mpd_path.name, e)

    logger.info("Parsed %d valid sets", len(valid_files))
    if not valid_files:
        return

    part_vocab, color_vocab = _build_vocab(all_bricks, top_k_parts=args.top_k_parts)
    logger.info("Vocabulary: %d parts, %d colors", len(part_vocab), len(color_vocab))

    sizes = [len(b) for b in all_bricks]
    max_bricks = int(np.percentile(sizes, 95))
    logger.info("N (95th percentile): %d bricks", max_bricks)

    # Convert to raw tensors to compute global scale
    raw_tensors = [_bricks_to_raw(b, part_vocab, color_vocab) for b in all_bricks]
    all_pos = np.concatenate([t["positions"].numpy() for t in raw_tensors])
    global_scale = float(np.percentile(np.abs(all_pos), 95))
    global_scale = max(global_scale, 1.0)
    logger.info("Global scale: %.1f LDU", global_scale)

    # Save vocabulary
    vocab = {
        "part_vocab": part_vocab,
        "color_vocab": color_vocab,
        "max_bricks": max_bricks,
        "global_scale": global_scale,
        "n_parts": len(part_vocab) + 1,
        "n_colors": len(color_vocab) + 1,
    }
    vocab_path = Path(args.out_dir) / f"vocab_{args.theme}.pt"
    torch.save(vocab, vocab_path)
    logger.info("Saved vocab to %s", vocab_path)

    # Clean existing set tensors so augmented and old unaugmented files don't coexist
    removed = sum(1 for f in out_dir.glob("*.pt") if f.name != vocab_path.name)
    for f in out_dir.glob("*.pt"):
        if f.name != vocab_path.name:
            f.unlink()
    if removed:
        logger.info("Removed %d stale .pt files from %s", removed, out_dir)

    # Pass 2: normalize, pad, augment (×8) and save per-set tensors
    saved = 0
    for mpd_path, raw in zip(valid_files, raw_tensors):
        n = min(raw["positions"].shape[0], max_bricks)

        # Non-padded base arrays (numpy for transform math)
        pos_np = (raw["positions"][:n] / global_scale).numpy().astype(np.float32)  # (n, 3)
        rot_np = raw["rot_ids"][:n].numpy()                                         # (n,) int64

        pad = max_bricks - n
        padding_mask = torch.zeros(max_bricks, dtype=torch.bool)
        padding_mask[n:] = True

        # Pad part/color once — they are augmentation-invariant
        part_ids = raw["part_ids"][:n]
        color_ids = raw["color_ids"][:n]
        if pad > 0:
            z_long = torch.zeros(pad, dtype=torch.long)
            part_ids = torch.cat([part_ids, z_long])
            color_ids = torch.cat([color_ids, z_long])

        for aug_idx, aug_mat in enumerate(_AUG_TRANSFORMS):
            aug_pos = torch.from_numpy((pos_np @ aug_mat.T).astype(np.float32))  # (n, 3)
            aug_rot = torch.from_numpy(_ROT_REMAP[aug_idx][rot_np])              # (n,) int64

            if pad > 0:
                aug_pos = torch.cat([aug_pos, torch.zeros(pad, 3)], dim=0)
                aug_rot = torch.cat([aug_rot, z_long])

            torch.save(
                {
                    "positions": aug_pos,
                    "rot_ids": aug_rot,
                    "part_ids": part_ids,
                    "color_ids": color_ids,
                    "padding_mask": padding_mask,
                    "n_bricks": n,
                },
                out_dir / f"{mpd_path.stem}_aug{aug_idx}.pt",
            )
        saved += 1

    logger.info(
        "Saved %d sets × 8 augmentations = %d tensors to %s",
        saved, saved * 8, out_dir,
    )


if __name__ == "__main__":
    main()

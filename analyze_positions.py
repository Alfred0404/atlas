"""Analyze the distribution of positions in the tokenized dataset."""

import os
import numpy as np
from collections import Counter

from src.config import Config

OFFSETS = Config.OFFSETS
POS_X_START = OFFSETS["positions_x"]  # 28
POS_Y_START = OFFSETS["positions_y"]  # 1028
POS_Z_START = OFFSETS["positions_z"]  # 2028
BINS = Config.NUM_BINS_PER_AXIS       # 1000
PRECISION = Config.PRECISION          # 2
MIN_POS = Config.MIN_POSITION         # -1000


def token_to_ldu(token, axis_offset):
    """Convert a position token back to LDU value."""
    bin_idx = token - axis_offset
    return MIN_POS + bin_idx * PRECISION


def main():
    npy_dir = Config.TOKENIZED_DATASET_DIR
    files = [f for f in os.listdir(npy_dir) if f.endswith(".npy")]
    print(f"Found {len(files)} tokenized files\n")

    all_x, all_y, all_z = [], [], []
    position_counter = Counter()  # (x_tok, y_tok, z_tok) -> count
    collision_counts = []  # per-set: how many duplicate positions

    for f in files:
        matrix = np.load(os.path.join(npy_dir, f))
        # Each row = [part_id, x, z, y, rotation, color]
        if matrix.ndim != 2 or matrix.shape[1] != 6:
            continue

        xs = matrix[:, 1]
        zs = matrix[:, 2]
        ys = matrix[:, 3]

        all_x.extend(xs.tolist())
        all_y.extend(ys.tolist())
        all_z.extend(zs.tolist())

        # Check collisions within this set
        positions = set()
        collisions = 0
        for i in range(len(matrix)):
            pos = (int(xs[i]), int(ys[i]), int(zs[i]))
            if pos in positions:
                collisions += 1
            positions.add(pos)
            position_counter[pos] += 1
        collision_counts.append(collisions)

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    total_bricks = len(all_x)

    print(f"Total bricks: {total_bricks}")
    print(f"Unique (x,y,z) positions: {len(position_counter)}")
    print()

    # === Per-axis stats (in LDU) ===
    for name, tokens, offset in [("X", all_x, POS_X_START),
                                  ("Y", all_y, POS_Y_START),
                                  ("Z", all_z, POS_Z_START)]:
        ldu = np.array([token_to_ldu(t, offset) for t in tokens])
        unique_bins = len(set(tokens.tolist()))
        print(f"--- {name} axis ---")
        print(f"  Range (LDU):  [{ldu.min():.0f}, {ldu.max():.0f}]")
        print(f"  Mean (LDU):   {ldu.mean():.1f}")
        print(f"  Std (LDU):    {ldu.std():.1f}")
        print(f"  Unique bins:  {unique_bins} / {BINS}")
        print(f"  Median (LDU): {np.median(ldu):.1f}")
        print()

    # === Concentration: what % of bricks fall in the top N positions? ===
    top_positions = position_counter.most_common(20)
    print("--- Top 20 most frequent positions ---")
    for pos, count in top_positions:
        x_ldu = token_to_ldu(pos[0], POS_X_START)
        y_ldu = token_to_ldu(pos[1], POS_Y_START)
        z_ldu = token_to_ldu(pos[2], POS_Z_START)
        pct = 100 * count / total_bricks
        print(f"  ({x_ldu:>5.0f}, {y_ldu:>5.0f}, {z_ldu:>5.0f}) LDU — {count:>5} times ({pct:.2f}%)")

    top10_count = sum(c for _, c in position_counter.most_common(10))
    top50_count = sum(c for _, c in position_counter.most_common(50))
    top100_count = sum(c for _, c in position_counter.most_common(100))
    print()
    print(f"Top 10 positions cover:  {100*top10_count/total_bricks:.1f}% of all bricks")
    print(f"Top 50 positions cover:  {100*top50_count/total_bricks:.1f}% of all bricks")
    print(f"Top 100 positions cover: {100*top100_count/total_bricks:.1f}% of all bricks")

    # === Collisions within sets ===
    collision_counts = np.array(collision_counts)
    sets_with_collisions = np.sum(collision_counts > 0)
    print()
    print(f"--- Collisions within sets ---")
    print(f"  Sets with duplicate positions: {sets_with_collisions} / {len(files)}")
    print(f"  Mean collisions per set:       {collision_counts.mean():.2f}")
    print(f"  Max collisions in a set:       {collision_counts.max()}")


if __name__ == "__main__":
    main()

"""Move flat graph_sets/*.npz into per-theme subdirectories.

Uses the existing dataset/mpd_files/<theme>/ structure as the source of truth.

Usage
-----
    python migrate_graph_sets.py [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MPD_DIR = Path("dataset/mpd_files")
GRAPH_DIR = Path("dataset/graph_sets")


def build_stem_to_theme(mpd_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for theme_dir in mpd_dir.iterdir():
        if not theme_dir.is_dir():
            continue
        for mpd_file in theme_dir.iterdir():
            if mpd_file.suffix.lower() in {".mpd", ".ldr"}:
                mapping[mpd_file.stem] = theme_dir.name
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stem_to_theme = build_stem_to_theme(MPD_DIR)

    npz_files = sorted(p for p in GRAPH_DIR.iterdir() if p.suffix == ".npz")
    print(f"Found {len(npz_files)} .npz files, {len(stem_to_theme)} MPD stems mapped across themes.")

    moved = skipped = unknown = 0
    for npz in npz_files:
        theme = stem_to_theme.get(npz.stem, "Unknown")
        dest = GRAPH_DIR / theme / npz.name
        if args.dry_run:
            print(f"  [{theme}] {npz.name}")
            moved += 1
            continue
        dest.parent.mkdir(exist_ok=True)
        shutil.move(str(npz), str(dest))
        moved += 1
        if theme == "Unknown":
            unknown += 1

    print(f"\nDone — moved: {moved}  unknown: {unknown}")


if __name__ == "__main__":
    main()

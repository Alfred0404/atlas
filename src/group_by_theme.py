"""Group MPD/LDR files in dataset/mpd_files/ into per-theme subdirectories.

Usage
-----
    python src/group_by_theme.py [--dry-run] [--workers N] [--merge]

--merge applies MERGE_GROUPS remapping during sorting (or post-hoc on existing dirs).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rebrick
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
rebrick.init(os.getenv("REBRICKABLE_API_KEY"))

MPD_DIR = Path("dataset/mpd_files")
SLEEP = 0.2  # seconds between the two API calls within one fetch

# Merge groups: target folder → list of source theme names to absorb into it.
# All sub-themes of LEGO Town that share the same structural vocabulary.
MERGE_GROUPS: dict[str, list[str]] = {
    "Town": [
        "Classic Town",
        "Town",
        "Town Jr.",
        "Traffic",
        "Police",
        "Airport",
        "Fire",
        "Harbor",
        "Gas Station",
    ],
}

# Flat lookup: source theme name → target theme name, built from MERGE_GROUPS.
_REMAP: dict[str, str] = {
    src: tgt
    for tgt, sources in MERGE_GROUPS.items()
    for src in sources
}

# Cache theme_id → name: many sets share the same theme, so get_theme is called
# once per unique theme rather than once per set.
_theme_name_cache: dict[int, str] = {}
_theme_name_lock = threading.Lock()


def _extract_set_number(filename: str) -> str:
    m = re.match(r"^(\d+(?:-\d+)?)", filename)
    return m.group(1) if m else ""


def _sanitize_dirname(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip()


def _fetch_theme(set_num: str) -> str | None:
    try:
        info = json.loads(rebrick.lego.get_set(set_num).read())
        theme_id = info["theme_id"]

        with _theme_name_lock:
            if theme_id in _theme_name_cache:
                return _theme_name_cache[theme_id]

        time.sleep(SLEEP)
        theme_info = json.loads(rebrick.lego.get_theme(theme_id).read())
        name = theme_info["name"]

        with _theme_name_lock:
            _theme_name_cache[theme_id] = name

        return name
    except Exception:
        return None


def _get_theme(set_num: str, merge: bool) -> str:
    theme = _fetch_theme(set_num)
    if theme is None and "-" not in set_num:
        # Rebrickable needs the full set number (e.g. "10001-1"), try with "-1" suffix
        theme = _fetch_theme(f"{set_num}-1")
    name = _sanitize_dirname(theme) if theme else "Unknown"
    if merge:
        name = _REMAP.get(name, name)
    return name


def apply_merges_posthoc(dry_run: bool) -> None:
    """Merge already-organized theme subdirs according to MERGE_GROUPS."""
    total_moved = 0
    for target, sources in MERGE_GROUPS.items():
        target_dir = MPD_DIR / target
        print(f"\nTarget: {target_dir.resolve()}")
        for source in sources:
            if source == target:
                continue
            source_dir = MPD_DIR / source
            if not source_dir.exists():
                print(f"  '{source}' — not found, skipping")
                continue
            files = [f for f in source_dir.iterdir() if f.is_file()]
            if not files:
                print(f"  '{source}' — empty, skipping")
                continue
            print(f"  Merging '{source}' ({len(files)} files) → '{target}'")
            for f in files:
                dest = target_dir / f.name
                if dry_run:
                    print(f"    [DRY] {f.name}")
                    continue
                target_dir.mkdir(exist_ok=True)
                if dest.exists():
                    print(f"    SKIP (exists): {f.name}")
                    continue
                shutil.move(str(f), str(dest))
                total_moved += 1
            if not dry_run and source_dir.exists() and not any(source_dir.iterdir()):
                source_dir.rmdir()
    if not dry_run:
        print(f"\nTotal files moved: {total_moved}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move MPD files into per-theme subdirectories.")
    parser.add_argument("--dry-run", action="store_true", help="Print moves without executing")
    parser.add_argument("--workers", type=int, default=5, help="Parallel API workers (default: 5)")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Apply MERGE_GROUPS: remap themes during sort, or reorganize existing subdirs if already sorted",
    )
    args = parser.parse_args()

    # If files are already in theme subdirs (no flat .mpd/.ldr at root), run post-hoc merge.
    flat_files = sorted(p for p in MPD_DIR.iterdir() if p.suffix.lower() in {".mpd", ".ldr"})
    if not flat_files:
        if args.merge:
            print(f"No flat files found in {MPD_DIR.resolve()} — applying post-hoc merge to existing subdirs.")
            apply_merges_posthoc(args.dry_run)
        else:
            print(f"No flat files found in {MPD_DIR.resolve()}. Already sorted? Use --merge to reorganize.")
        return

    print(f"Found {len(flat_files)} files in {MPD_DIR.resolve()}")

    unique_sets = sorted({_extract_set_number(f.name) for f in flat_files} - {""})
    print(f"Fetching themes for {len(unique_sets)} unique set numbers with {args.workers} workers...")
    if args.merge:
        print(f"Merge enabled: {len(_REMAP)} theme names will be remapped.")

    set_theme: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_get_theme, sn, args.merge): sn for sn in unique_sets}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching themes"):
            set_num = futures[future]
            set_theme[set_num] = future.result()

    counts: dict[str, int] = {}
    for f in flat_files:
        set_num = _extract_set_number(f.name)
        theme = set_theme.get(set_num, "Unknown")
        counts[theme] = counts.get(theme, 0) + 1
        dest = MPD_DIR / theme / f.name

        if args.dry_run:
            continue

        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            print(f"SKIP (exists): {dest}")
            continue
        shutil.move(str(f), str(dest))

    print("\nSummary:")
    for theme, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {theme}: {count} files")


if __name__ == "__main__":
    main()

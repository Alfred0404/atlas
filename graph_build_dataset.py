"""Preprocess MPD files into graph .npz files for the GraphTransformer.

Usage
-----
    python graph_build_dataset.py [--mpd-dir PATH] [--output-dir PATH] [-j N]

Steps
-----
1. Load technic blacklist and filter MPD files.
2. Pass 1 — scan all files to collect the full part/color vocabulary.
3. Pass 1.5 — pre-warm PartDatabase cache for all known part IDs, save to disk.
4. Pass 2 — parallel processing: N workers each load the pre-built cache and
             call process_mpd() on their assigned files.
5. Save vocabulary mappings alongside the dataset (graph_vocab.pt).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.parser import MPDParser
from src.geometry.graph_builder import process_mpd
from src.geometry.lego_part import PartDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MPD_DIR = Path("dataset/mpd_files")
OUTPUT_DIR = Path("dataset/graph_sets")
BLACKLIST = Path("dataset/technic_blacklist.txt")
VOCAB_PATH = Path("dataset/graph_vocab.pt")
DB_CACHE = Path("dataset/part_db_cache.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_blacklist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(path.read_text().strip().splitlines())


def _extract_set_number(filename: str) -> str:
    m = re.match(r"^(\d+)", filename)
    return m.group(1) if m else ""


def _filter_files(mpd_dir: Path, blacklist: set[str]) -> list[Path]:
    files = sorted(p for p in mpd_dir.iterdir() if p.suffix.lower() in {".mpd", ".ldr"})
    before = len(files)
    files = [f for f in files if _extract_set_number(f.name) not in blacklist]
    logger.info(
        "Blacklist removed %d / %d files — %d remain",
        before - len(files),
        before,
        len(files),
    )
    return files


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

# Module-level so it's picklable for multiprocessing.
_worker_db: PartDatabase | None = None
_worker_part_vocab: dict[str, int] | None = None
_worker_color_vocab: dict[int, int] | None = None
_worker_save_compressed: bool = True


def _init_worker(
    cache_path: str,
    part_vocab: dict[str, int],
    color_vocab: dict[int, int],
    save_compressed: bool,
) -> None:
    """Pool initializer: load shared read-only state once per worker."""
    global _worker_db, _worker_part_vocab, _worker_color_vocab, _worker_save_compressed
    _worker_db = PartDatabase()
    cp = Path(cache_path)
    if cp.exists():
        _worker_db.load_cache(cp)
    _worker_part_vocab = part_vocab
    _worker_color_vocab = color_vocab
    _worker_save_compressed = save_compressed


def _process_one(args: tuple) -> tuple[str, bool]:
    """Process a single MPD file. Returns (stem, success)."""
    mpd_path, out_path = args
    mpd_path = Path(mpd_path)
    out_path = Path(out_path)
    try:
        data = process_mpd(
            mpd_path,
            _worker_db,
            _worker_part_vocab,
            _worker_color_vocab,
        )
        if data is None:
            return mpd_path.stem, False
        if _worker_save_compressed:
            np.savez_compressed(str(out_path), **data)
        else:
            np.savez(str(out_path), **data)
        return mpd_path.stem, True
    except Exception as exc:
        logger.debug("Failed %s: %s", mpd_path.name, exc)
        return mpd_path.stem, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph dataset from MPD files.")
    parser.add_argument("--mpd-dir", default=str(MPD_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--blacklist", default=str(BLACKLIST))
    parser.add_argument("--vocab-path", default=str(VOCAB_PATH))
    parser.add_argument("--db-cache", default=str(DB_CACHE))
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of parallel workers (default: cpu_count-1)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        default=True,
        help="Reprocess files even if .npz already exists",
    )
    parser.add_argument(
        "--no-compress",
        dest="save_compressed",
        action="store_false",
        default=True,
        help="Save .npz without compression for faster write speed",
    )
    args = parser.parse_args()

    mpd_dir = Path(args.mpd_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blacklist = _load_blacklist(Path(args.blacklist))
    files = _filter_files(mpd_dir, blacklist)
    if not files:
        logger.error("No MPD files found after filtering.")
        return

    # ---- Pass 1: vocabulary -------------------------------------------------
    vocab_path = Path(args.vocab_path)
    if vocab_path.exists():
        logger.info("Loading existing vocabulary from %s", vocab_path)
        saved = torch.load(vocab_path, weights_only=True)
        part_vocab = saved["part_vocab"]
        color_vocab = saved["color_vocab"]
    else:
        logger.info("Pass 1 — scanning vocabulary across %d files ...", len(files))
        all_parts: set[str] = set()
        all_colors: set[int] = set()
        for path in tqdm(files, desc="Vocabulary scan", unit="file"):
            try:
                p = MPDParser(str(path))
                raw = p.parse(next(iter(p._submodels)))
                for b in raw:
                    all_parts.add(b.brick_id)
                    all_colors.add(int(b.color))
            except Exception:
                continue

        part_vocab = {p: i + 1 for i, p in enumerate(sorted(all_parts))}
        color_vocab = {c: i + 1 for i, c in enumerate(sorted(all_colors))}
        torch.save({"part_vocab": part_vocab, "color_vocab": color_vocab}, vocab_path)
        logger.info(
            "Vocabulary: %d parts, %d colors — saved to %s",
            len(part_vocab),
            len(color_vocab),
            vocab_path,
        )

    logger.info("Vocabulary: %d parts, %d colors", len(part_vocab), len(color_vocab))

    # ---- Pass 1.5: warm PartDatabase cache ----------------------------------
    db_cache_path = Path(args.db_cache)
    if not db_cache_path.exists():
        logger.info(
            "Pass 1.5 — pre-warming PartDatabase for %d unique parts ...",
            len(part_vocab),
        )
        db = PartDatabase()
        for part_id in tqdm(
            sorted(part_vocab.keys()), desc="Warming part cache", unit="part"
        ):
            db.get_or_default(part_id)
        db.save_cache(db_cache_path)
        logger.info("Part cache saved to %s", db_cache_path)
    else:
        logger.info("Using existing part cache at %s", db_cache_path)

    # ---- Pass 2: parallel graph building ------------------------------------
    work_items = []
    for path in files:
        out_path = output_dir / (path.stem + ".npz")
        if args.skip_existing and out_path.exists():
            continue
        work_items.append((str(path), str(out_path)))

    n_skip = len(files) - len(work_items)
    compression_label = "compressed" if args.save_compressed else "uncompressed"
    logger.info(
        "Pass 2 — %d files to process (%d already exist), using %d workers (%s output) ...",
        len(work_items),
        n_skip,
        args.jobs,
        compression_label,
    )

    if not work_items:
        logger.info("Nothing to do.")
        return

    ok = fail = 0
    with mp.Pool(
        processes=args.jobs,
        initializer=_init_worker,
        initargs=(str(db_cache_path), part_vocab, color_vocab, args.save_compressed),
    ) as pool:
        chunksize = max(8, len(work_items) // max(1, args.jobs * 8))
        for _, success in tqdm(
            pool.imap_unordered(_process_one, work_items, chunksize=chunksize),
            total=len(work_items),
            desc="Building graphs",
            unit="file",
        ):
            if success:
                ok += 1
            else:
                fail += 1

    logger.info(
        "Done — ok=%d  skipped=%d  failed=%d  total=%d",
        ok,
        n_skip,
        fail,
        len(files),
    )
    logger.info("Next: python graph_train_model.py --graph-dir %s", output_dir)


if __name__ == "__main__":
    main()

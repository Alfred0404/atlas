"""
Scraper for the LDraw Official Model Repository (OMR).
Downloads .mpd files from https://library.ldraw.org/omr/sets/
Skips sets that already exist in the local dataset directory.
"""

import sys
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Add src to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.logging import setup_logging
    from src.config import Config
else:
    from ..utils.logging import setup_logging
    from ..config import Config

logger = setup_logging()

BASE_URL = "https://library.ldraw.org"
SETS_URL = f"{BASE_URL}/omr/sets/"


def get_existing_set_numbers(directory: str) -> set[str]:
    """Scan existing files and extract set numbers (digits at start of filename)."""
    numbers = set()
    for filename in os.listdir(directory):
        match = re.match(r"^(\d+)", filename)
        if match:
            numbers.add(match.group(1))
    return numbers


def scrape_set_list(page: int) -> list[dict]:
    """Scrape one page of the OMR set list.

    Returns:
        List of dicts with keys: set_number, name, page_url
    """
    url = f"{SETS_URL}?page={page}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    sets = []
    for row in soup.select("tr.fi-ta-row"):
        # Set number
        number_cell = row.select_one("td.fi-ta-cell-number")
        if not number_cell:
            continue
        link = number_cell.select_one("a")
        if not link:
            continue

        set_number = link.get_text(strip=True)
        page_url = link.get("href", "")
        if not page_url.startswith("http"):
            page_url = BASE_URL + page_url

        # Set name
        name_cell = row.select_one("td.fi-ta-cell-name")
        name = name_cell.get_text(strip=True) if name_cell else set_number

        sets.append({
            "set_number": set_number,
            "name": name,
            "page_url": page_url,
        })

    return sets


def get_mpd_links(set_page_url: str) -> list[str]:
    """Fetch a set's page and extract all .mpd download links."""
    response = requests.get(set_page_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".mpd"):
            if not href.startswith("http"):
                href = BASE_URL + href
            links.append(href)

    return links


def download_file(url: str, save_path: str) -> bool:
    """Download a file from a URL."""
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def main():
    download_dir = Config.RAW_DATASET_DIR
    os.makedirs(download_dir, exist_ok=True)

    # Collect existing set numbers to skip
    existing = get_existing_set_numbers(download_dir)
    logger.info(f"Found {len(existing)} existing sets in {download_dir}")

    # Scrape all pages
    all_sets = []
    page = 1
    while True:
        logger.info(f"Scraping page {page}...")
        try:
            sets = scrape_set_list(page)
        except Exception as e:
            logger.error(f"Error scraping page {page}: {e}")
            break

        if not sets:
            break

        all_sets.extend(sets)
        page += 1
        time.sleep(0.3)

    logger.info(f"Found {len(all_sets)} sets on OMR")

    # Filter out sets we already have
    to_download = []
    for s in all_sets:
        # Extract base number (e.g. "10001" from "10001-1")
        base_number = s["set_number"].split("-")[0]
        if base_number in existing:
            continue
        to_download.append(s)

    logger.info(f"Skipping {len(all_sets) - len(to_download)} already downloaded sets")
    logger.info(f"Downloading {len(to_download)} new sets")

    success = 0
    failed = 0

    for i, s in enumerate(to_download, 1):
        logger.info(f"[{i}/{len(to_download)}] {s['set_number']} - {s['name']}")

        try:
            mpd_links = get_mpd_links(s["page_url"])
        except Exception as e:
            logger.error(f"  Error fetching set page: {e}")
            failed += 1
            continue

        if not mpd_links:
            logger.warning(f"  No .mpd files found")
            failed += 1
            continue

        for mpd_url in mpd_links:
            filename = mpd_url.split("/")[-1]
            save_path = os.path.join(download_dir, filename)

            if os.path.exists(save_path):
                logger.info(f"  Skipping (exists): {filename}")
                continue

            if download_file(mpd_url, save_path):
                logger.info(f"  Downloaded: {filename}")
                success += 1
            else:
                failed += 1

        time.sleep(0.5)

    logger.info("=" * 60)
    logger.info(f"Done! Downloaded: {success}, Failed: {failed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

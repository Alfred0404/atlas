"""
Script to download all .mpd and .ldr files from seymouria.pl
Downloads official LEGO set files in LDraw format.
"""

import sys
from pathlib import Path
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Add src to path for direct execution
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))

from ..utils.logging import setup_logging

logger = setup_logging()


def download_file(url: str, save_path: str) -> bool:
    """Download a file from a URL to the specified path.
    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path to save the downloaded file.
    Returns:
        bool: True if download was successful, False otherwise.
    """

    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return True

    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def get_all_file_links(base_url: str):
    """
    Scrape the webpage and extract all .mpd and .ldr file links.
    Args:
        base_url (str): The URL of the webpage to scrape.
    Returns:
        list of tuples: A list of (filename, file_url) tuples.
    """

    try:
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        file_links = []

        # Find all anchor tags that link to .mpd or .ldr files
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)

            # Look for .mpd or .ldr files
            if (
                ".mpd" in href.lower() or ".ldr" in href.lower()
            ) and "OfficialLegoSets_LDR" in href:
                filename = text
                # Convert relative path to absolute URL
                file_url = urljoin(base_url, href)

                file_links.append((filename, file_url))

        return file_links

    except Exception as e:
        logger.error(f"Error fetching page: {e}")
        return []


def main():
    """Main function to download all files."""
    base_url = "https://www.seymouria.pl/Download/official-lego-sets-ldr.php"
    download_dir = os.path.join(os.getcwd(), "downloaded_lego_files")

    logger.info("Fetching file list from seymouria.pl...")
    file_links = get_all_file_links(base_url)

    if not file_links:
        logger.warning("No files found. The page structure might have changed.")
        return

    logger.info(f"Found {len(file_links)} files to download.")
    logger.info(f"Download directory: {download_dir}")

    # Create download directory
    os.makedirs(download_dir, exist_ok=True)

    # Download each file
    success_count = 0
    failed_count = 0

    for i, (filename, file_url) in enumerate(file_links, 1):
        save_path = os.path.join(download_dir, filename)

        # Skip if file already exists
        if os.path.exists(save_path):
            logger.info(
                f"[{i}/{len(file_links)}] Skipping (already exists): {filename}"
            )
            success_count += 1
            continue

        logger.info(f"[{i}/{len(file_links)}] Downloading: {filename}")

        if download_file(file_url, save_path):
            success_count += 1
            logger.info(f"  ✓ Saved to: {save_path}")
        else:
            failed_count += 1
            logger.warning(f"  ✗ Failed to download")

        # Be polite - add a small delay between downloads
        if i < len(file_links):
            time.sleep(0.5)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"Download complete!")
    logger.info(f"Successfully downloaded: {success_count}/{len(file_links)}")
    if failed_count > 0:
        logger.warning(f"Failed: {failed_count}/{len(file_links)}")
    logger.info(f"Files saved to: {download_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

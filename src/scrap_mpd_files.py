"""
Script to download all .mpd and .ldr files from seymouria.pl
Downloads official LEGO set files in LDraw format.
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote


def download_file(url, save_path):
    """Download a file from a URL to the specified path."""
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
        print(f"Error downloading {url}: {e}")
        return False


def get_all_file_links(base_url):
    """
    Scrape the webpage and extract all .mpd and .ldr file links.
    Returns a list of tuples: (filename, file_url)
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
        print(f"Error fetching page: {e}")
        return []


def main():
    """Main function to download all files."""
    base_url = "https://www.seymouria.pl/Download/official-lego-sets-ldr.php"
    download_dir = os.path.join(os.getcwd(), "downloaded_lego_files")

    print("Fetching file list from seymouria.pl...")
    file_links = get_all_file_links(base_url)

    if not file_links:
        print("No files found. The page structure might have changed.")
        return

    print(f"Found {len(file_links)} files to download.")
    print(f"Download directory: {download_dir}\n")

    # Create download directory
    os.makedirs(download_dir, exist_ok=True)

    # Download each file
    success_count = 0
    failed_count = 0

    for i, (filename, file_url) in enumerate(file_links, 1):
        save_path = os.path.join(download_dir, filename)

        # Skip if file already exists
        if os.path.exists(save_path):
            print(f"[{i}/{len(file_links)}] Skipping (already exists): {filename}")
            success_count += 1
            continue

        print(f"[{i}/{len(file_links)}] Downloading: {filename}")

        if download_file(file_url, save_path):
            success_count += 1
            print(f"  ✓ Saved to: {save_path}")
        else:
            failed_count += 1
            print(f"  ✗ Failed to download")

        # Be polite - add a small delay between downloads
        if i < len(file_links):
            time.sleep(0.5)

    # Summary
    print("\n" + "=" * 60)
    print(f"Download complete!")
    print(f"Successfully downloaded: {success_count}/{len(file_links)}")
    print(f"Failed: {failed_count}/{len(file_links)}")
    print(f"Files saved to: {download_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

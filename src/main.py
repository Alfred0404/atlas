"""
Main entry point for running the ATLAS dataset builder.

This script properly handles imports and runs the DatasetBuilder to process
LEGO MPD files into tokenized datasets.
"""

import sys
from pathlib import Path

# Add src to path so absolute imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.builder import DatasetBuilder
from src.config import Config


def main():
    """Run the dataset builder to process MPD files."""
    print("=" * 60)
    print("Starting ATLAS Dataset Builder")
    print("=" * 60)

    # Initialize the dataset builder
    dataset_builder = DatasetBuilder(Config.ATLAS_CONFIG_PATH)

    # Process all MPD files in the dataset directory
    dataset_builder.process_dataset(max_files=10)

    print("=" * 60)
    print("Dataset processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

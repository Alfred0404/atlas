"""
Example demonstrating the clean separation between MPDParser and DatasetBuilder.

MPDParser: Responsible for parsing MPD files
DatasetBuilder: Responsible for transforming parsed data into training-ready format
"""

from MPDParser import MPDParser
from src.DatasetBuilder import DatasetBuilder
import numpy as np

# Step 1: Parse MPD file (Parser responsibility)
print("=" * 60)
print("STEP 1: PARSING MPD FILE")
print("=" * 60)
parser = MPDParser("./mpd_files/test.mpd")
raw_data = parser.parse("4484 - Main Model.ldr\n")
print(f"✓ Parsed {len(raw_data)} bricks")
print(f"✓ Raw data contains: brick_id, world_matrix, color")
print()

# Step 2: Transform data (DatasetBuilder responsibility)
print("=" * 60)
print("STEP 2: TRANSFORMING DATA")
print("=" * 60)
builder = DatasetBuilder(metadata_path="./metadata.json")

# Set the raw data from parser
builder.raw_data = raw_data

# Apply transformations
print("- Centering model around origin...")
builder.center_around_origin()

print("- Converting to quaternion representation...")
builder.to_quat_representation()

print("- Calculating max brick distance...")
max_dist = builder.calculate_max_brick_distance()

print("- Creating ID mappings...")
brick_mapping = builder.map_brick_ids()
color_mapping = builder.map_brick_colors()

print("- Creating processed data...")
id_mapping = {"brick_id_mapping": brick_mapping, "color_mapping": color_mapping}
processed_data = builder.use_id_mapping(id_mapping)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"✓ Quaternion data: {len(builder.quat_data)} bricks")
print(f"✓ Processed data: {len(processed_data)} bricks")
print(f"✓ Max brick distance: {max_dist:.2f}")
print(f"✓ Unique brick types: {len(brick_mapping)}")
print(f"✓ Unique colors: {len(color_mapping)}")
print()

# Example: Show first brick in each format
if raw_data and builder.quat_data and processed_data:
    print("=" * 60)
    print("EXAMPLE: First brick in different formats")
    print("=" * 60)

    print("\n1. Raw format (from parser):")
    print(f"   brick_id: {raw_data[0].brick_id}")
    print(f"   color: {raw_data[0].color}")
    print(f"   world_matrix shape: {raw_data[0].world_matrix.shape}")

    print("\n2. Quaternion format (from builder):")
    print(f"   brick_id: {builder.quat_data[0].brick_id}")
    print(f"   position: {builder.quat_data[0].position}")
    print(f"   rotation_quat: {builder.quat_data[0].rotation_quat}")
    print(f"   color: {builder.quat_data[0].color}")

    print("\n3. Processed format (training-ready):")
    print(f"   brick_idx: {processed_data[0].brick_idx}")
    print(f"   position: {processed_data[0].position}")
    print(f"   rotation_quat: {processed_data[0].rotation_quat}")
    print(f"   color_idx: {processed_data[0].color_idx}")
    print()

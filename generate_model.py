import os
import torch
import numpy as np
from datetime import datetime

from src.config import Config
from src.core.tokenizer import AtlasTokenizer
from src.core.vocabulary import VocabularyManager
from src.data.parser import RawBrickData
from src.file_io.mpd_writer import write_mpd_file
from src.maths.rotations import generate_chiral_rotation_matrices
from src.model import ModelConfig, ATLASTransformer, Generator
from src.utils.logging import setup_logging

logger = setup_logging()


def tokens_to_raw_bricks(bricks: list[dict], vocab_manager: VocabularyManager) -> list[RawBrickData]:
    """Convert decoded brick dicts (token IDs) back to RawBrickData objects."""
    tokenizer = AtlasTokenizer()
    offsets = vocab_manager.get_offsets()

    # Build reverse lookups: token_id → original value
    # Parts: index → part_id string
    parts_vocab = vocab_manager.get_all_parts()  # {"3001.dat": 0, ...}
    token_to_part = {idx + offsets["parts"]: pid for pid, idx in parts_vocab.items()}

    # Colors: index → color int
    colors_vocab = vocab_manager.get_all_colors()  # {"16": 0, ...}
    token_to_color = {idx + offsets["colors"]: int(cid) for cid, idx in colors_vocab.items()}

    # Rotations: index → 3x3 matrix
    rotation_matrices = generate_chiral_rotation_matrices()

    raw_bricks = []
    for brick in bricks:
        # Part ID
        part_id = token_to_part.get(brick["part_id"], "3001.dat")

        # Position (decode bin tokens back to real coordinates)
        x = tokenizer.bin_id_to_position(brick["x"], "x")
        y = tokenizer.bin_id_to_position(brick["y"], "y")
        z = tokenizer.bin_id_to_position(brick["z"], "z")

        # Rotation (token → matrix index → 3x3 matrix)
        rot_idx = brick["rotation"] - offsets["rotations"]
        rot_idx = max(0, min(rot_idx, len(rotation_matrices) - 1))
        rotation = rotation_matrices[rot_idx]

        # Color
        color = token_to_color.get(brick["color"], 16)  # default to color 16

        # Build 4x4 world matrix
        world_matrix = np.eye(4)
        world_matrix[:3, :3] = rotation
        world_matrix[:3, 3] = [x, y, z]

        raw_bricks.append(RawBrickData(
            brick_id=part_id,
            world_matrix=world_matrix,
            color=color,
        ))

    return raw_bricks


def main():
    """Load a trained checkpoint, generate a LEGO model and export it as an MPD file."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load vocabulary
    vocab_manager = VocabularyManager(Config.ATLAS_CONFIG_PATH)
    vocab_size = vocab_manager.get_vocab_size()
    offsets = vocab_manager.get_offsets()

    # Load checkpoint and rebuild model
    checkpoint_path = "./checkpoints/atlas_transformer.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: ModelConfig = checkpoint["config"]
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")

    model = ATLASTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Generate
    generator = Generator(model, config, device=device)
    tokens = generator.generate(max_bricks=400, temperature=0.8, top_k=50)
    bricks = generator.decode_sequence(tokens)
    logger.info(f"Generated {len(bricks)} bricks")

    # Convert to RawBrickData and write MPD
    raw_bricks = tokens_to_raw_bricks(bricks, vocab_manager)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./generated_sets/generated_{timestamp}.mpd"
    write_mpd_file(output_path, raw_bricks, model_name=f"atlas_{timestamp}")
    logger.info(f"MPD file written to {output_path}")


if __name__ == "__main__":
    main()

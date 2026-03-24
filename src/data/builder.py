import re
import sys
from pathlib import Path
from typing import NamedTuple, List, Set, Optional
import numpy as np
from tqdm import tqdm

# Add src to path for direct execution
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))

from .adjacency import sort_bricks_by_adjacency
from .augmentation import AugmentationConfig, generate_augmented_variants
from .parser import MPDParser, RawBrickData
from ..core.vocabulary import VocabularyManager
from ..core.tokenizer import AtlasTokenizer
from ..maths.transforms import (
    batch_get_rotation_matrices,
    get_position_from_world_matrix,
    get_rotation_matrix_from_world_matrix,
)
from ..utils.logging import setup_logging
from ..config import Config

logger = setup_logging()


class TokenizedBrickData(NamedTuple):
    """Training ready data with brick_id and color mapped to integers"""

    brick_idx: int  # mapped unique integer for brick_id
    position: np.ndarray  # shape (3,) [x,y,z]
    rotation_idx: int  # rotation token ID from vocabulary
    color_idx: int  # mapped unique integer for color


class DatasetBuilder:
    """Build and process LEGO datasets from MPD files.

    This class handles the complete pipeline from raw MPD files to processed
    tensor data, managing vocabulary through VocabularyManager, and the tokenization process through AtlasTokenizer.
    """

    def __init__(
        self,
        atlas_config_path: str,
        augmentation_config: AugmentationConfig = None,
    ):
        """
        Initialize the DatasetBuilder.

        Args:
            atlas_config_path: Path to the atlas_config.json file.
            augmentation_config: Optional augmentation config. None disables augmentation.
        """
        self.vocab_manager = VocabularyManager(atlas_config_path)
        self.tokenizer = AtlasTokenizer()
        self.aug_config = augmentation_config
        self.all_brick_ids: Set[str] = set()
        self.all_colors: Set[int] = set()

        # Data storage for processing individual files
        self.raw_data: List[RawBrickData] = []
        self.tokenized_data: List[TokenizedBrickData] = []
        self.final_tensor = None

    @staticmethod
    def _extract_set_number(filename: str) -> str:
        """Extract the leading numeric set number from a filename."""
        match = re.match(r"^(\d+)", filename)
        return match.group(1) if match else ""

    @staticmethod
    def _load_blacklist() -> Set[str]:
        """Load set numbers to exclude from the blacklist file."""
        path = Path(Config.TECHNIC_BLACKLIST_PATH)
        if not path.exists():
            return set()
        return set(path.read_text().strip().splitlines())

    def process_dataset(self, max_files: Optional[int] = None) -> None:
        """
        Process all MPD files in the raw dataset directory.

        Two-pass pipeline:
        1. Parse all files and collect vocabulary (parts + colors).
        2. Tokenize all files with the frozen vocabulary.

        Args:
            max_files: Maximum number of files to process. If None, process all files.
        """
        logger.info("Starting dataset processing...\n")

        all_files = list(Path(Config.RAW_DATASET_DIR).glob("*.mpd"))

        if not all_files:
            logger.error(f"No MPD files found in directory: {Config.RAW_DATASET_DIR}")
            return

        # Filter out blacklisted sets (Technic, Bionicle, Hero Factory)
        blacklist = self._load_blacklist()
        if blacklist:
            before = len(all_files)
            all_files = [f for f in all_files if self._extract_set_number(f.name) not in blacklist]
            skipped = before - len(all_files)
            logger.info(f"Blacklist: skipped {skipped} files ({len(blacklist)} set numbers loaded)")

        if max_files is not None:
            all_files = all_files[:max_files]
            logger.info(f"Processing limited to {len(all_files)} files.\n")

        # --- Pass 1: collect vocabulary and cache parsed data ---
        logger.info("Pass 1: Collecting vocabulary...")
        parsed_cache: dict[Path, List[RawBrickData]] = {}
        for mpd_file in tqdm(all_files, desc="Scanning vocabulary", unit="file"):
            raw_data = self._parse_and_transform(mpd_file)
            if raw_data is None:
                continue
            parsed_cache[mpd_file] = raw_data
            self.all_brick_ids.update(brick.brick_id for brick in raw_data)
            self.all_colors.update(brick.color for brick in raw_data)

        # Build complete vocabulary (colors first, then parts) and save once
        self.vocab_manager.add_colors(self.all_colors)
        self.vocab_manager.add_parts(self.all_brick_ids)
        self.vocab_manager.save()

        logger.info("Vocabulary summary:")
        logger.info(f"Total unique parts: {self.vocab_manager.get_parts_count()}")
        logger.info(f"Total unique colors: {self.vocab_manager.get_colors_count()}")
        logger.info(f"Total vocabulary size: {self.vocab_manager.get_vocab_size()}")

        # Load frozen vocabulary into tokenizer
        self.tokenizer.load_vocabulary(Config.ATLAS_CONFIG_PATH)

        # --- Pass 2: tokenize and save (using cached parse results) ---
        logger.info("Pass 2: Tokenizing dataset...")
        rng = np.random.default_rng(
            self.aug_config.seed if self.aug_config else 42
        )

        for mpd_file in tqdm(parsed_cache, desc="Tokenizing", unit="file"):
            raw_data = parsed_cache[mpd_file]

            # Generate augmented variants (or just original if no config)
            if self.aug_config:
                variants = generate_augmented_variants(raw_data, self.aug_config)
            else:
                variants = [("", raw_data)]

            for suffix, variant_bricks in variants:
                # Identity: already sorted and centered from pass 1.
                # Geometric variants (rot/mirror): distances between bricks
                # are invariant under isometries, so BFS order is the same.
                # The bricks are already in the right order — just re-center.
                # Permutation variants (_pN): need randomized BFS for a
                # different ordering.
                is_permutation = "_p" in suffix

                if is_permutation:
                    self.raw_data = sort_bricks_by_adjacency(
                        variant_bricks, rng=rng
                    )
                    self.center_around_origin()
                elif suffix == "":
                    self.raw_data = variant_bricks
                else:
                    # Geometric variant: skip re-sort, just re-center
                    self.raw_data = variant_bricks
                    self.center_around_origin()

                self.tokenize_brick_data()
                self._to_tensor()
                output_path = (
                    Path(Config.TOKENIZED_DATASET_DIR)
                    / f"{mpd_file.stem}{suffix}.npy"
                )
                self.save_dataset(str(output_path))

        logger.info("Dataset processing complete.\n")

    def _parse_and_transform(self, mpd_file_path) -> Optional[List[RawBrickData]]:
        """Parse an MPD file and return transformed raw brick data, or None on failure."""
        parser = MPDParser(str(mpd_file_path))

        if not parser._submodels:
            logger.warning(f"No submodels found in {mpd_file_path}. Skipping.")
            return None

        first_submodel_key = list(parser._submodels.keys())[0]
        self.raw_data = parser.flatten(first_submodel_key, np.eye(4))

        if not self.raw_data:
            logger.warning(f"No raw data extracted from {mpd_file_path}. Skipping.")
            return None

        self.raw_data = sort_bricks_by_adjacency(self.raw_data)
        self.center_around_origin()

        return self.raw_data

    def center_around_origin(self) -> None:
        """
        Normalize model space by centering X and Z axes on 10 LDU grid.

        Aligns the model base (lowest point) to Y = 0. This ensures consistent
        positioning across all models.
        """
        if not self.raw_data:
            logger.warning("No raw data to center.")
            return

        # 1. Extraction des positions actuelles
        positions = np.array(
            [
                get_position_from_world_matrix(brick.world_matrix)
                for brick in self.raw_data
            ]
        )

        # 2. Calcul des limites (Bounding Box)
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)

        # 3. Calcul du centre théorique pour X et Z
        center_x = (min_coords[0] + max_coords[0]) / 2
        center_z = (min_coords[2] + max_coords[2]) / 2

        # 4. Snapping sur la grille (10 LDU pour X/Z, 8 LDU pour Y)
        # Pour Y, on prend le point le plus bas (max_coords[1] en LDraw car Y positif descend)
        # On le snappe à 8 LDU pour rester propre
        snapped_offset = np.array(
            [
                np.round(center_x / 10.0) * 10.0,
                np.round(max_coords[1] / 8.0) * 8.0,
                np.round(center_z / 10.0) * 10.0,
            ]
        )

        # 5. Mise à jour des matrices mondiales
        for brick in self.raw_data:
            brick.world_matrix[:3, 3] -= snapped_offset

    def collect_brick_ids(self) -> Set[str]:
        """
        Collect unique brick IDs from current raw data.

        Returns:
            Set of unique brick identifiers from the current file.
        """
        if not self.raw_data:
            logger.warning("No raw data to collect brick IDs.")
            return set()

        # Get unique brick IDs
        unique_brick_ids = set(brick.brick_id for brick in self.raw_data)
        logger.debug(
            f"Collected {len(unique_brick_ids)} unique brick IDs from current file."
        )
        return unique_brick_ids

    def collect_brick_colors(self) -> Set[int]:
        """
        Collect unique brick colors from current raw data.

        Returns:
            Set of unique color codes from the current file.
        """
        if not self.raw_data:
            logger.warning("No raw data to collect brick colors.")
            return set()

        # Get unique colors
        unique_colors = set(brick.color for brick in self.raw_data)
        logger.debug(f"Collected {len(unique_colors)} unique colors from current file.")

        return unique_colors

    def _update_vocabulary(self, new_brick_ids: Set[str], new_colors: Set[int]) -> None:
        """
        Update vocabulary with new parts and colors.

        This method adds newly encountered parts and colors to the atlas_config.json
        vocabulary using the VocabularyManager.

        Args:
            new_brick_ids: Set of new brick identifiers to add.
            new_colors: Set of new color codes to add.
        """
        if new_brick_ids:
            self.vocab_manager.add_parts(new_brick_ids)
            logger.debug(f"Added {len(new_brick_ids)} new parts to vocabulary")

        if new_colors:
            self.vocab_manager.add_colors(new_colors)
            logger.debug(f"Added {len(new_colors)} new colors to vocabulary")

    def tokenize_brick_data(self) -> None:
        """
        Tokenize all brick attributes using vectorized batch operations.

        Extracts positions and rotations from world matrices in batch,
        then tokenizes everything via numpy array ops. Produces both
        self.tokenized_data (for backward compat) and self.final_tensor.
        """
        if not self.raw_data:
            logger.warning("No raw data to tokenize.")
            return

        n = len(self.raw_data)

        # Stack all world matrices into (N, 4, 4)
        world_matrices = np.array([b.world_matrix for b in self.raw_data])

        # Batch extract positions (N, 3) and rotation matrices (N, 3, 3)
        positions = world_matrices[:, :3, 3]
        rotation_matrices = batch_get_rotation_matrices(world_matrices)

        # Batch tokenize
        pos_tokens = self.tokenizer.batch_positions_to_bin_ids(positions)
        rot_tokens = self.tokenizer.batch_rotation_matrices_to_tokens(rotation_matrices)
        brick_tokens = self.tokenizer.batch_brick_ids_to_tokens(
            [b.brick_id for b in self.raw_data]
        )
        color_tokens = self.tokenizer.batch_color_ids_to_tokens(
            [str(b.color) for b in self.raw_data]
        )

        # Assemble (N, 6) tensor: [brick_idx, x, y, z, rotation_idx, color_idx]
        self.final_tensor = np.column_stack([
            brick_tokens, pos_tokens, rot_tokens, color_tokens
        ])

        # Backward compat: populate tokenized_data for tests
        self.tokenized_data = [
            TokenizedBrickData(
                brick_idx=int(brick_tokens[i]),
                position=pos_tokens[i],
                rotation_idx=int(rot_tokens[i]),
                color_idx=int(color_tokens[i]),
            )
            for i in range(n)
        ]

        logger.info(f"Tokenized {n} bricks (vectorized)")
        logger.debug(f"Final tensor shape: {self.final_tensor.shape}")

    def _to_tensor(self) -> None:
        """No-op: tensor is now built directly in tokenize_brick_data()."""
        pass

    def save_dataset(self, output_path: str) -> None:
        """
        Save the final tensor dataset to a .npy file.

        Args:
            output_path: Path where the .npy file will be saved.
        """
        if self.final_tensor is not None:
            np.save(output_path, self.final_tensor)
            logger.info(f"Dataset saved to {output_path}\n")
        else:
            logger.warning("Final tensor is empty. Nothing to save.\n")


if __name__ == "__main__":
    dataset_builder = DatasetBuilder(Config.ATLAS_CONFIG_PATH)
    # Process only 10 files for testing
    dataset_builder.process_dataset(max_files=10)

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.core.vocabulary import VocabularyManager
from src.data.sequence_dataset import SequenceDataset, collate_fn
from src.model import ModelConfig, ATLASTransformer, Trainer
from src.utils.logging import setup_logging

logger = setup_logging()


def main():
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Vocabulary
    vocab_manager = VocabularyManager(Config.ATLAS_CONFIG_PATH)
    vocab_size = vocab_manager.get_vocab_size()
    offsets = vocab_manager.get_offsets()
    logger.info(f"Vocabulary size (from config): {vocab_size}")

    # Dataset
    dataset = SequenceDataset(
        npy_dir=Config.TOKENIZED_DATASET_DIR,
        max_bricks=200,
        max_files=100_000,
    )
    logger.info(f"Dataset size: {len(dataset)} sequences")

    # Scan dataset: expand vocab_size if needed, drop files with invalid tokens
    logger.info("Scanning dataset for token range...")
    max_token = 0
    bad_indices = []
    for i in range(len(dataset)):
        seq = dataset[i]
        if seq.min().item() < 0:
            bad_indices.append(i)
            continue
        val = seq.max().item()
        if val > max_token:
            max_token = val
    if bad_indices:
        logger.warning(
            f"Dropping {len(bad_indices)} sequences with negative tokens "
            f"(positions out of bounds)"
        )
        # Remove bad files from dataset (iterate in reverse to preserve indices)
        for i in sorted(bad_indices, reverse=True):
            dataset.npy_files.pop(i)
    logger.info(f"Max token in dataset: {max_token}, clean sequences: {len(dataset)}")
    if max_token >= vocab_size:
        vocab_size = max_token + 1
        logger.warning(
            f"Expanding vocab_size to {vocab_size} to cover all tokens"
        )

    # Config
    config = ModelConfig(vocab_size=vocab_size, offsets=offsets)

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Model
    model = ATLASTransformer(config)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Train
    trainer = Trainer(model, train_loader, config, device=device)
    trainer.train()


if __name__ == "__main__":
    main()

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.config import load_config, repo_root
from src.data.split import load_split
from src.data.text import clean_caption, load_captions
from src.data.vocab import Vocabulary


class Flickr8kDataset(Dataset):
    """Bir örnek, bir görüntünün bir altyazısıdır.

    token_ids uzunluğu 34'tür: <start>, kelimeler, <end>, sağda <pad>.
    Önbellek burada okunmaz. Eğitim, image_id ile önbellek tensörünü eşler.
    """

    def __init__(self, split: str, config_path: Path | None = None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split train, val veya test olmalı")
        config = load_config(config_path)
        root = repo_root()
        self.split = split
        self.images_dir = root / config["images_dir"]
        self.vocab = Vocabulary.load(root / config["vocab_path"])
        captions = load_captions(root / config["captions_path"])
        image_ids = load_split(root / config["split_path"])[split]
        self.samples = [
            (image_id, caption)
            for image_id in image_ids
            for caption in captions[image_id]
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image_id, raw_caption = self.samples[index]
        caption = clean_caption(raw_caption)
        token_ids = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        return {
            "image_id": image_id,
            "image_path": str(self.images_dir / image_id),
            "caption": caption,
            "token_ids": token_ids,
        }


def collate_captions(batch: list[dict]) -> dict:
    return {
        "image_id": [item["image_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "token_ids": torch.stack([item["token_ids"] for item in batch]),
    }


def build_dataloader(
    split: str,
    batch_size: int = 32,
    shuffle: bool | None = None,
    seed: int | None = None,
    num_workers: int = 0,
    config_path: Path | None = None,
) -> DataLoader:
    config = load_config(config_path)
    if shuffle is None:
        shuffle = split == "train"
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(config["seed"] if seed is None else seed))
    dataset = Flickr8kDataset(split, config_path=config_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_captions,
        generator=generator,
    )

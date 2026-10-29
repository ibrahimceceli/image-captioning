from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.cache import FeatureStore, cache_paths
from src.data.config import load_config, repo_root
from src.data.split import load_split
from src.data.text import clean_caption, load_captions
from src.data.vocab import Vocabulary


def _open_store(config: dict, image_ids: list[str], features_dir: Path | None) -> FeatureStore:
    store = FeatureStore(*cache_paths(config, features_dir))
    store.require(image_ids)
    return store


def _check_eval_split(split: str, allow_test: bool) -> None:
    if split not in {"train", "val", "test"}:
        raise ValueError("split train, val veya test olmalı")
    if split == "test" and not allow_test:
        raise ValueError("Test kümesi metrik için açılmaz. Final tablo için allow_test=True verin")


class Flickr8kDataset(Dataset):
    """Bir örnek, bir görüntünün bir altyazısıdır.

    token_ids uzunluğu 34'tür: <start>, kelimeler, <end>, sağda <pad>.
    use_features=True ise örnek önbellekten float32 features [2048, 7, 7] da taşır.
    """

    def __init__(
        self,
        split: str,
        config_path: Path | None = None,
        use_features: bool = False,
        features_dir: Path | None = None,
    ):
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
        self.features = _open_store(config, image_ids, features_dir) if use_features else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image_id, raw_caption = self.samples[index]
        caption = clean_caption(raw_caption)
        token_ids = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        item = {
            "image_id": image_id,
            "image_path": str(self.images_dir / image_id),
            "caption": caption,
            "token_ids": token_ids,
        }
        if self.features is not None:
            item["features"] = self.features[image_id]
        return item


class ImageFeatureDataset(Dataset):
    """Görüntü başına bir örnek: image_id ve features. Doğrulama çözümlemesi içindir."""

    def __init__(
        self,
        split: str = "val",
        config_path: Path | None = None,
        features_dir: Path | None = None,
        allow_test: bool = False,
    ):
        _check_eval_split(split, allow_test)
        config = load_config(config_path)
        self.split = split
        self.image_ids = load_split(repo_root() / config["split_path"])[split]
        self.features = _open_store(config, self.image_ids, features_dir)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict:
        image_id = self.image_ids[index]
        return {"image_id": image_id, "features": self.features[image_id]}


def load_references(
    split: str = "val",
    config_path: Path | None = None,
    allow_test: bool = False,
) -> dict[str, list[str]]:
    """{image_id: 5 temiz altyazı}. Metrik referansı, clean_caption ile aynı temizlik."""
    _check_eval_split(split, allow_test)
    config = load_config(config_path)
    root = repo_root()
    captions = load_captions(root / config["captions_path"])
    image_ids = load_split(root / config["split_path"])[split]
    return {image_id: [clean_caption(text) for text in captions[image_id]] for image_id in image_ids}


def collate_captions(batch: list[dict]) -> dict:
    collated = {
        "image_id": [item["image_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "token_ids": torch.stack([item["token_ids"] for item in batch]),
    }
    if "features" in batch[0]:
        collated["features"] = torch.stack([item["features"] for item in batch])
    return collated


def collate_images(batch: list[dict]) -> dict:
    return {
        "image_id": [item["image_id"] for item in batch],
        "features": torch.stack([item["features"] for item in batch]),
    }


def build_dataloader(
    split: str,
    batch_size: int = 32,
    shuffle: bool | None = None,
    seed: int | None = None,
    num_workers: int = 0,
    config_path: Path | None = None,
    use_features: bool = False,
    features_dir: Path | None = None,
) -> DataLoader:
    config = load_config(config_path)
    if shuffle is None:
        shuffle = split == "train"
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(config["seed"] if seed is None else seed))
    dataset = Flickr8kDataset(
        split,
        config_path=config_path,
        use_features=use_features,
        features_dir=features_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_captions,
        generator=generator,
    )


def build_eval_dataloader(
    split: str = "val",
    batch_size: int = 64,
    num_workers: int = 0,
    config_path: Path | None = None,
    features_dir: Path | None = None,
    allow_test: bool = False,
) -> DataLoader:
    """Görüntü başına bir örnek, sıra sabit. load_references ile image_id üzerinden eşlenir."""
    dataset = ImageFeatureDataset(
        split,
        config_path=config_path,
        features_dir=features_dir,
        allow_test=allow_test,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_images,
    )

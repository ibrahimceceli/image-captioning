"""Öznitelik önbelleğini okur. Dosyalar src.data.cache_features ile yazılır."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from src.data.config import repo_root

FEATURE_SHAPE = (2048, 7, 7)


def meta_path(features_path: Path) -> Path:
    """resnet50_layer4.npy -> resnet50_layer4.meta.json"""
    return features_path.with_suffix(".meta.json")


def cache_paths(config: dict, features_dir: Path | None = None) -> tuple[Path, Path]:
    """Config yollarını çözer. features_dir verilirse dosya adları o klasörde aranır."""
    for key in ("features_path", "features_index_path"):
        if key not in config:
            raise KeyError(f"Config alanı eksik: {key}")
    features_path = repo_root() / config["features_path"]
    index_path = repo_root() / config["features_index_path"]
    if features_dir is not None:
        features_path = Path(features_dir) / features_path.name
        index_path = Path(features_dir) / index_path.name
    return features_path, index_path


def load_index(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Önbellek indeksi yok: {path}. Önce python -m src.data.cache_features çalıştırın")
    return json.loads(path.read_text(encoding="utf-8"))


class FeatureStore:
    """image_id ile float32 [2048, 7, 7] döndürür.

    Memmap ilk erişimde açılır ve pickle edilmez; her DataLoader worker'ı kendi
    kopyasını açar.
    """

    def __init__(self, features_path: Path, index_path: Path):
        if not features_path.is_file():
            raise FileNotFoundError(f"Önbellek dosyası yok: {features_path}. Önce python -m src.data.cache_features çalıştırın")
        self.features_path = features_path
        self.index = load_index(index_path)
        self._array: np.ndarray | None = None
        shape = self.array.shape
        if shape != (len(self.index), *FEATURE_SHAPE):
            raise ValueError(f"Önbellek şekli {shape}, indeks {len(self.index)} satır bekliyor")

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self.features_path, mmap_mode="r")
        return self._array

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_array"] = None
        return state

    def __contains__(self, image_id: str) -> bool:
        return image_id in self.index

    def require(self, image_ids: list[str]) -> None:
        missing = [image_id for image_id in image_ids if image_id not in self.index]
        if missing:
            raise KeyError(f"Önbellekte {len(missing)} görüntü eksik, ilki: {missing[0]}")

    def __getitem__(self, image_id: str) -> Tensor:
        row = self.array[self.index[image_id]]
        return torch.from_numpy(np.asarray(row, dtype=np.float32))

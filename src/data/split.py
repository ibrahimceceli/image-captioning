import json
import random
from pathlib import Path

SPLIT_COUNTS = {"train": 6000, "val": 1000, "test": 1000}
SPLIT_METHOD = (
    "Görüntü adları sıralanır, random.Random(seed).shuffle uygulanır, "
    "ilk 6000 train, sonraki 1000 val, sonraki 1000 test olur. "
    "Kalan görüntüler unused içine yazılır ve eğitime girmez. "
    "Küme listeleri alfabetik kaydedilir."
)


def build_split(image_ids: list[str], seed: int) -> dict:
    ids = sorted(set(image_ids))
    needed = sum(SPLIT_COUNTS.values())
    if len(ids) < needed:
        raise ValueError(f"En az {needed} görüntü gerekir, bulunan: {len(ids)}")

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    train = shuffled[:6000]
    val = shuffled[6000:7000]
    test = shuffled[7000:8000]
    unused = shuffled[8000:]
    parts = {
        "train": train,
        "val": val,
        "test": test,
        "unused": unused,
    }
    _assert_partition(ids, parts)
    return {
        "seed": seed,
        "method": SPLIT_METHOD,
        "counts": {name: len(items) for name, items in parts.items()},
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "unused": sorted(unused),
    }


def save_split(split: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")


def load_split(path: Path) -> dict:
    split = json.loads(path.read_text(encoding="utf-8"))
    for name in ("train", "val", "test", "unused"):
        if name not in split:
            raise KeyError(f"Bölme dosyasında {name} yok: {path}")
    return split


def _assert_partition(image_ids: list[str], parts: dict[str, list[str]]) -> None:
    for name, expected in SPLIT_COUNTS.items():
        if len(parts[name]) != expected:
            raise ValueError(f"{name} kümesi {expected} olmalı, bulunan: {len(parts[name])}")
    seen: set[str] = set()
    for name in ("train", "val", "test", "unused"):
        overlap = seen & set(parts[name])
        if overlap:
            raise ValueError(f"{name} kümesi başka kümeyle görüntü paylaşıyor")
        seen.update(parts[name])
    if seen != set(image_ids):
        raise ValueError("Bölme, görüntü listesinin tamamını kapsamıyor")

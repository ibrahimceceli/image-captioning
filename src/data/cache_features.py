"""ResNet50 layer4 önbelleğini yazar: python -m src.data.cache_features

Çıktılar (out-dir içinde, adlar configs/base.yaml'dan):
  resnet50_layer4.npy        fp16 [N, 2048, 7, 7], satırlar image_id sırasıyla
  resnet50_layer4_index.json {image_id: satır}
  resnet50_layer4.meta.json  ağırlık, dtype, şekil, sürümler
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from src.data.cache import FEATURE_SHAPE, cache_paths, meta_path
from src.data.config import load_config, repo_root
from src.data.features import WEIGHTS_NAME, build_transform, extract_batch, load_encoder, open_image


class ImageFiles(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(open_image(self.paths[index]))


def list_images(images_dir: Path, limit: int | None = None) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Görüntü klasörü yok: {images_dir}")
    paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() == ".jpg")
    if not paths:
        raise ValueError(f"Klasörde .jpg yok: {images_dir}")
    return paths[:limit] if limit else paths


def parse_args() -> argparse.Namespace:
    config = load_config()
    default_out = cache_paths(config)[0].parent
    parser = argparse.ArgumentParser(description="ResNet50 layer4 öznitelik önbelleği")
    parser.add_argument("--images-dir", type=Path, default=repo_root() / config["images_dir"])
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Yalnız ilk N görüntü (deneme için)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true", help="GPU'da autocast kullanma")
    return parser.parse_args()


def cache_features(args: argparse.Namespace) -> dict:
    config = load_config()
    features_path, index_path = cache_paths(config, features_dir=args.out_dir)
    paths = list_images(args.images_dir, args.limit)
    shape = (len(paths), *FEATURE_SHAPE)
    amp = not args.no_amp and torch.device(args.device).type == "cuda"

    features_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float16, shape=shape)
    encoder = load_encoder(args.device)
    loader = DataLoader(
        ImageFiles(paths, build_transform(config)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    start, row = time.time(), 0
    for batch in loader:
        features = extract_batch(batch, encoder, args.device, amp=amp)
        output[row : row + len(features)] = features.cpu().numpy().astype(np.float16)
        row += len(features)
        print(f"{row}/{len(paths)} görüntü, {time.time() - start:.0f} sn", flush=True)
    output.flush()
    del output

    index = {path.name: i for i, path in enumerate(paths)}
    index_path.write_text(json.dumps(index, indent=0), encoding="utf-8")
    meta = {
        "weights": WEIGHTS_NAME,
        "layer": "layer4 (avgpool ve fc atıldı)",
        "dtype": "float16",
        "shape": list(shape),
        "amp": amp,
        "resize_short_edge": int(config["resize_short_edge"]),
        "image_size": int(config["image_size"]),
        "imagenet_mean": config["imagenet_mean"],
        "imagenet_std": config["imagenet_std"],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(args.device),
        "seconds": round(time.time() - start, 1),
    }
    meta_path(features_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    meta = cache_features(parse_args())
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

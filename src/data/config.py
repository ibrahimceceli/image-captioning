from pathlib import Path

import yaml

REQUIRED_KEYS = (
    "seed",
    "min_word_freq",
    "max_length",
    "special_tokens",
    "pad_side",
    "images_dir",
    "captions_path",
    "split_path",
    "vocab_path",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    config_path = path or (repo_root() / "configs" / "base.yaml")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise KeyError(f"Eksik config alanları: {', '.join(missing)}")
    return config

"""Data exports; load tensor/image dependencies only when they are used."""

from importlib import import_module

from src.data.text import clean_caption, load_captions
from src.data.vocab import Vocabulary

_LAZY_MODULES = {
    "FeatureStore": "cache",
    **dict.fromkeys(("Flickr8kDataset", "ImageFeatureDataset", "build_dataloader",
                     "build_eval_dataloader", "load_references"), "dataset"),
    **dict.fromkeys(("build_transform", "extract_batch", "extract_image", "load_encoder"), "features"),
}


def __getattr__(name):
    if name not in _LAZY_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_LAZY_MODULES[name]}"), name)
    globals()[name] = value
    return value

__all__ = [
    "FeatureStore",
    "Flickr8kDataset",
    "ImageFeatureDataset",
    "Vocabulary",
    "build_dataloader",
    "build_eval_dataloader",
    "build_transform",
    "clean_caption",
    "extract_batch",
    "extract_image",
    "load_captions",
    "load_encoder",
    "load_references",
]

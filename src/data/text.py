import csv
import re
from collections import defaultdict
from pathlib import Path

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")


def clean_caption(text: str) -> str:
    """Küçük harfe çevirir ve noktalama işaretlerini siler."""
    lowered = text.lower()
    without_punct = _NON_ALNUM.sub("", lowered)
    return _SPACES.sub(" ", without_punct).strip()


def load_captions(path: Path) -> dict[str, list[str]]:
    captions: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "image" not in fields or "caption" not in fields:
            raise ValueError("captions.txt başlıkları image ve caption olmalı")
        for row in reader:
            captions[row["image"]].append(row["caption"])
    return dict(captions)

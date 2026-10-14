import json
from collections import Counter
from pathlib import Path

from src.data.text import clean_caption

SPECIAL_ORDER = ("<pad>", "<start>", "<end>", "<unk>")


class Vocabulary:
    def __init__(self, itos: list[str], min_word_freq: int, max_length: int):
        if tuple(itos[:4]) != SPECIAL_ORDER:
            raise ValueError("Sözlük <pad> <start> <end> <unk> sırasında başlamalı")
        if max_length < 2:
            raise ValueError("max_length en az 2 olmalı")
        self.itos = list(itos)
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.min_word_freq = int(min_word_freq)
        self.max_length = int(max_length)
        self.pad_idx = self.stoi["<pad>"]
        self.start_idx = self.stoi["<start>"]
        self.end_idx = self.stoi["<end>"]
        self.unk_idx = self.stoi["<unk>"]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(
        cls,
        captions: list[str],
        min_word_freq: int,
        max_length: int,
    ) -> tuple["Vocabulary", Counter]:
        counts: Counter[str] = Counter()
        for caption in captions:
            counts.update(clean_caption(caption).split())
        words = [
            word
            for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_word_freq
        ]
        vocab = cls(list(SPECIAL_ORDER) + words, min_word_freq, max_length)
        return vocab, counts

    def encode(self, text: str) -> list[int]:
        words = clean_caption(text).split()[: self.max_length - 2]
        ids = [self.start_idx]
        ids.extend(self.stoi.get(word, self.unk_idx) for word in words)
        ids.append(self.end_idx)
        ids.extend([self.pad_idx] * (self.max_length - len(ids)))
        return ids

    def decode(self, ids) -> str:
        words = []
        for raw_id in ids:
            token = self.itos[int(raw_id)]
            if token == "<end>":
                break
            if token in ("<pad>", "<start>"):
                continue
            words.append(token)
        return " ".join(words)

    def save(self, path: Path) -> None:
        payload = {
            "min_word_freq": self.min_word_freq,
            "max_length": self.max_length,
            "special_tokens": list(SPECIAL_ORDER),
            "itos": self.itos,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["itos"], payload["min_word_freq"], payload["max_length"])

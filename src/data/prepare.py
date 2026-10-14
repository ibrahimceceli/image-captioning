import json
from pathlib import Path

import torch

from src.data.config import load_config, repo_root
from src.data.dataset import build_dataloader
from src.data.eda import plot_sentence_lengths, plot_word_frequencies
from src.data.split import build_split, load_split, save_split
from src.data.text import clean_caption, load_captions
from src.data.vocab import Vocabulary


def _lengths(captions: list[str]) -> list[int]:
    return [len(clean_caption(caption).split()) for caption in captions]


def _percentile(sorted_values: list[int], q: float) -> int:
    index = round(q * (len(sorted_values) - 1))
    return sorted_values[index]


def _assert_pad_right(token_ids: torch.Tensor, pad_idx: int, end_idx: int) -> None:
    for row in token_ids.tolist():
        end_pos = row.index(end_idx)
        if any(token == pad_idx for token in row[: end_pos + 1]):
            raise AssertionError("Pad, <end> token'ından önce duruyor")
        if any(token != pad_idx for token in row[end_pos + 1 :]):
            raise AssertionError("Pad sağa yaslı değil")


def _check_batch(split_name: str, split: dict, batch_size: int) -> None:
    loader = build_dataloader(split_name, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    ids = batch["token_ids"]
    if ids.shape != (batch_size, loader.dataset.vocab.max_length):
        raise AssertionError(f"{split_name} batch şekli {tuple(ids.shape)}")
    if ids.dtype != torch.long:
        raise AssertionError("token_ids long olmalı")
    if not torch.all(ids[:, 0] == loader.dataset.vocab.start_idx):
        raise AssertionError("Dizi <start> ile başlamıyor")
    _assert_pad_right(ids, loader.dataset.vocab.pad_idx, loader.dataset.vocab.end_idx)
    forbidden = set(split["val"] + split["test"] + split["unused"]) if split_name == "train" else set()
    if split_name == "val":
        forbidden = set(split["train"] + split["test"] + split["unused"])
    if set(batch["image_id"]) & forbidden:
        raise AssertionError(f"{split_name} batch'i başka kümeden görüntü içeriyor")
    rows = zip(batch["image_id"], batch["caption"], ids, batch["image_path"])
    for image_id, caption, row, image_path in rows:
        if loader.dataset.vocab.encode(caption) != row.tolist():
            raise AssertionError(f"Kodlama uyuşmuyor: {image_id}")
        if not Path(image_path).is_file():
            raise AssertionError(f"Görüntü dosyası yok: {image_id}")


def prepare() -> dict:
    config = load_config()
    root = repo_root()
    captions = load_captions(root / config["captions_path"])
    images_dir = root / config["images_dir"]
    image_ids = sorted(path.name for path in images_dir.iterdir() if path.suffix.lower() == ".jpg")
    if set(image_ids) != set(captions):
        raise ValueError("Görüntü dosyaları ile captions.txt eşleşmiyor")
    if any(len(items) != 5 for items in captions.values()):
        raise ValueError("Her görüntüde 5 altyazı olmalı")

    sample = captions["1000268201_693b08cb0e.jpg"][0]
    expected = "a child in a pink dress is climbing up a set of stairs in an entry way"
    if clean_caption(sample) != expected:
        raise AssertionError("Metin temizliği beklenen cümleyi vermedi")
    if clean_caption("A Dog's, toy!") != "a dogs toy":
        raise AssertionError("Noktalama temizliği beklenen sonucu vermedi")

    split = build_split(image_ids, seed=int(config["seed"]))
    split_path = root / config["split_path"]
    save_split(split, split_path)
    if load_split(split_path)["counts"] != split["counts"]:
        raise AssertionError("Bölme dosyası okunamadı")

    train_captions = [caption for image_id in split["train"] for caption in captions[image_id]]
    vocab, counts = Vocabulary.build(
        train_captions,
        min_word_freq=int(config["min_word_freq"]),
        max_length=int(config["max_length"]),
    )
    vocab_path = root / config["vocab_path"]
    vocab.save(vocab_path)
    loaded = Vocabulary.load(vocab_path)
    if loaded.itos != vocab.itos:
        raise AssertionError("Sözlük dosyası okunamadı")

    lengths = _lengths(train_captions)
    ordered = sorted(lengths)
    max_content = int(config["max_length"]) - 2
    n_kept = sum(1 for count in counts.values() if count >= vocab.min_word_freq)
    summary = {
        "n_images": len(image_ids),
        "captions_per_image": 5,
        "split_counts": split["counts"],
        "seed": int(config["seed"]),
        "train_captions": len(train_captions),
        "vocab_size": len(vocab),
        "special_tokens": 4,
        "kept_words": n_kept,
        "dropped_words": len(counts) - n_kept,
        "min_word_freq": vocab.min_word_freq,
        "max_length": vocab.max_length,
        "max_content_tokens": max_content,
        "length_min": ordered[0],
        "length_median": _percentile(ordered, 0.5),
        "length_mean": round(sum(lengths) / len(lengths), 2),
        "length_p95": _percentile(ordered, 0.95),
        "length_max": ordered[-1],
        "truncated_captions": sum(1 for length in lengths if length > max_content),
        "empty_captions": sum(1 for length in lengths if length == 0),
    }

    eda_dir = root / "results" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    plot_sentence_lengths(lengths, max_content, eda_dir / "sentence_lengths.png")
    plot_word_frequencies(counts, vocab.min_word_freq, eda_dir / "word_frequencies.png")
    summary_path = eda_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _check_batch("train", split, batch_size=32)
    _check_batch("val", split, batch_size=32)
    return summary


def main() -> None:
    summary = prepare()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""GloVe vektörlerinden sözlüğe hizalı bir embedding matrisi kurar.

    matrix, stats = load_glove_matrix(Path("data/glove/glove.6B.300d.txt"), vocab, 300)

Satır i, vocab.itos[i] kelimesinin vektörüdür:
- kelime GloVe'da varsa GloVe vektörü,
- yoksa (ve <start>, <end>, <unk> için) N(0, σ) rastgele; σ sözlükte bulunan
  GloVe vektörlerinin tüm bileşenlerinin standart sapmasıdır, üreteç seed'i
  sabittir (varsayılan 42), yani aynı dosya + sözlük her zaman aynı matrisi verir,
- <pad> satırı sıfırdır.

Dosya satır satır okunur; yalnız sözlükte geçen kelimelerin sayıları ayrıştırılır,
400k satırın tamamı belleğe alınmaz.
"""

from pathlib import Path

import torch
from torch import Tensor

from src.data.vocab import SPECIAL_ORDER, Vocabulary

MISSING_EXAMPLES = 30


def load_glove_matrix(path: Path, vocab: Vocabulary, dim: int, seed: int = 42) -> tuple[Tensor, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GloVe dosyası bulunamadı: {path}")
    wanted = {word: index for index, word in enumerate(vocab.itos) if word not in SPECIAL_ORDER}
    found: dict[int, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            word, _, rest = line.rstrip("\n").partition(" ")
            index = wanted.get(word)
            if index is None or index in found:
                continue
            values = rest.split(" ")
            if len(values) != dim:
                raise ValueError(
                    f"GloVe boyutu uyuşmuyor: {path.name} satır {line_no} '{word}' {len(values)} boyutlu, beklenen {dim}"
                )
            found[index] = [float(value) for value in values]

    if found:
        vectors = torch.tensor(list(found.values()), dtype=torch.float32)
        init_std = float(vectors.std())
    else:
        vectors, init_std = None, 1.0

    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(len(vocab), dim, generator=generator) * init_std
    if vectors is not None:
        matrix[list(found.keys())] = vectors
    matrix[vocab.pad_idx] = 0.0

    missing = [word for word, index in wanted.items() if index not in found]
    stats = {
        "glove_path": path.as_posix(),
        "dim": dim,
        "vocab_size": len(vocab),
        "words": len(wanted),
        "covered": len(found),
        "missing": len(missing),
        "coverage": round(len(found) / len(wanted), 4) if wanted else 0.0,
        "special_tokens": "<pad> sıfır; <start>/<end>/<unk> rastgele",
        "random_init": f"N(0, {init_std:.4f}), seed {seed}",
        "init_std": round(init_std, 6),
        "missing_examples": missing[:MISSING_EXAMPLES],
    }
    return matrix, stats

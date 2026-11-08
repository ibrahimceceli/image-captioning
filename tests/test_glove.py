"""GloVe embedding başlatma testleri (src/data/glove.py, src/train.py apply_pretrained_embeddings).

Gerçek GloVe dosyası gerekmez; tmp klasörde küçük sahte bir dosya yazılır.
"""

from pathlib import Path

import pytest
import torch

from src.data.glove import load_glove_matrix
from src.data.vocab import SPECIAL_ORDER, Vocabulary
from src.train import apply_pretrained_embeddings, build_model, load_run_config

ROOT = Path(__file__).resolve().parents[1]
DIM = 4
WORDS = ["dog", "runs", "grass", "zzqx"]  # zzqx GloVe dosyasında yok
GLOVE = {
    "the": [9.0, 9.0, 9.0, 9.0],          # sözlükte yok, okunmamalı
    "dog": [0.1, 0.2, 0.3, 0.4],
    "runs": [-0.5, 0.5, -0.5, 0.5],
    "grass": [1.0, -1.0, 2.0, -2.0],
}


def _vocab() -> Vocabulary:
    return Vocabulary(list(SPECIAL_ORDER) + WORDS, min_word_freq=1, max_length=10)


def _glove_file(tmp_path: Path) -> Path:
    path = tmp_path / "fake_glove.txt"
    path.write_text(
        "".join(f"{word} {' '.join(str(v) for v in vec)}\n" for word, vec in GLOVE.items()), encoding="utf-8"
    )
    return path


def test_matrix_rows_match_glove(tmp_path):
    vocab = _vocab()
    matrix, stats = load_glove_matrix(_glove_file(tmp_path), vocab, DIM)
    assert matrix.shape == (len(vocab), DIM)
    for word in ("dog", "runs", "grass"):
        assert torch.allclose(matrix[vocab.stoi[word]], torch.tensor(GLOVE[word]))
    assert stats["words"] == 4 and stats["covered"] == 3 and stats["missing"] == 1
    assert stats["coverage"] == 0.75
    assert stats["missing_examples"] == ["zzqx"]
    found = torch.tensor([GLOVE[w] for w in ("dog", "runs", "grass")])
    assert stats["init_std"] == pytest.approx(float(found.std()), abs=1e-6)


def test_pad_is_zero_and_specials_random(tmp_path):
    vocab = _vocab()
    matrix, _ = load_glove_matrix(_glove_file(tmp_path), vocab, DIM)
    assert torch.count_nonzero(matrix[vocab.pad_idx]) == 0
    for index in (vocab.start_idx, vocab.end_idx, vocab.unk_idx, vocab.stoi["zzqx"]):
        assert torch.count_nonzero(matrix[index]) == DIM


def test_missing_words_are_deterministic(tmp_path):
    vocab = _vocab()
    path = _glove_file(tmp_path)
    torch.manual_seed(0)
    first, _ = load_glove_matrix(path, vocab, DIM)
    torch.manual_seed(123)  # global RNG sonucu etkilememeli
    second, _ = load_glove_matrix(path, vocab, DIM)
    assert torch.equal(first, second)
    other_seed, _ = load_glove_matrix(path, vocab, DIM, seed=7)
    assert not torch.equal(first[vocab.stoi["zzqx"]], other_seed[vocab.stoi["zzqx"]])


def test_glove_does_not_touch_global_rng(tmp_path):
    torch.manual_seed(42)
    load_glove_matrix(_glove_file(tmp_path), _vocab(), DIM)
    after = torch.rand(3)
    torch.manual_seed(42)
    assert torch.equal(after, torch.rand(3))


def test_dimension_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="boyut"):
        load_glove_matrix(_glove_file(tmp_path), _vocab(), DIM + 1)


def _config(embeddings=None, embedding_dim=DIM) -> dict:
    config = {"name": "t", "seed": 42,
              "model": {"type": "lstm", "embedding_dim": embedding_dim, "hidden_dim": 8, "num_layers": 1, "dropout": 0.0}}
    if embeddings is not None:
        config["embeddings"] = embeddings
    return config


def test_apply_initializes_embedding(tmp_path):
    vocab = _vocab()
    config = _config({"path": str(_glove_file(tmp_path)), "freeze": False})
    torch.manual_seed(42)
    model = build_model(config, vocab)
    stats = apply_pretrained_embeddings(model, config, vocab, ROOT, tmp_path / "res")
    expected, _ = load_glove_matrix(_glove_file(tmp_path), vocab, DIM, seed=42)
    assert torch.equal(model.embedding.weight.detach(), expected)
    assert model.embedding.weight.requires_grad
    assert stats["coverage"] == 0.75
    assert (tmp_path / "res" / "embedding_stats.json").is_file()


def test_apply_freeze(tmp_path):
    vocab = _vocab()
    config = _config({"path": str(_glove_file(tmp_path)), "freeze": True})
    model = build_model(config, vocab)
    apply_pretrained_embeddings(model, config, vocab, ROOT)
    assert not model.embedding.weight.requires_grad


def test_apply_dimension_mismatch(tmp_path):
    vocab = _vocab()
    config = _config({"path": str(_glove_file(tmp_path)), "dim": 300})
    with pytest.raises(ValueError, match="embedding_dim"):
        apply_pretrained_embeddings(build_model(config, vocab), config, vocab, ROOT)


def test_no_embeddings_section_is_noop(tmp_path):
    """Alan yokken model ve global RNG, yardımcı hiç çağrılmamış gibi kalır."""
    vocab = _vocab()
    config = _config()
    torch.manual_seed(42)
    reference = build_model(config, vocab).state_dict()
    reference_next = torch.rand(3)
    torch.manual_seed(42)
    model = build_model(config, vocab)
    assert apply_pretrained_embeddings(model, config, vocab, ROOT, tmp_path / "res") is None
    assert torch.equal(torch.rand(3), reference_next)
    for key, value in model.state_dict().items():
        assert torch.equal(value, reference[key])
    assert not (tmp_path / "res").exists()


@pytest.mark.parametrize("name", ["model1", "model2", "model1_emb300", "model1_glove300"])
def test_configs_do_not_pass_embeddings_to_model(name):
    """build_model model bölümünü kwarg yapar; embeddings bölümü oraya girmemeli."""
    config = load_run_config(ROOT / "configs" / f"{name}.yaml")
    assert "pretrained_embeddings" not in config["model"]
    assert "embeddings" not in config["model"]
    if name in ("model1", "model2", "model1_emb300"):
        assert "embeddings" not in config


def test_glove_config_differs_from_control_only_in_embeddings():
    control = load_run_config(ROOT / "configs" / "model1_emb300.yaml")
    glove = load_run_config(ROOT / "configs" / "model1_glove300.yaml")
    base = load_run_config(ROOT / "configs" / "model1.yaml")
    assert glove.pop("embeddings")["freeze"] is False
    assert {k: v for k, v in glove.items() if k != "name"} == {k: v for k, v in control.items() if k != "name"}
    assert control["model"]["embedding_dim"] == 300 and base["model"]["embedding_dim"] == 256
    control["model"]["embedding_dim"] = 256
    assert {k: v for k, v in control.items() if k != "name"} == {k: v for k, v in base.items() if k != "name"}

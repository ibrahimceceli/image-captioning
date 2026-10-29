"""Sanity checks for Model 1 (CaptionLSTM) using synthetic features.

Runs with pytest or directly from the repo root:
``PYTHONPATH=. python tests/test_lstm.py``.
No feature cache is needed; captions are real Flickr8k sentences encoded
with the shared vocabulary.
"""

from pathlib import Path

import torch
from torch import nn

from src.data.vocab import Vocabulary
from src.models.lstm import CaptionLSTM

VOCAB_PATH = Path(__file__).resolve().parents[1] / "src" / "data" / "artifacts" / "vocab.json"
CAPTIONS = [
    "A little girl climbing into a wooden playhouse .",
    "A black dog and a spotted dog are fighting",
    "A man lays on a bench while his dog sits by him .",
    "Two young guys with shaggy hair look at their hands while hanging out in the yard .",
]


def _vocab() -> Vocabulary:
    return Vocabulary.load(VOCAB_PATH)


def _loss(model, features, token_ids, pad_idx):
    logits, attention = model(features, token_ids[:, :-1])
    assert attention is None
    targets = token_ids[:, 1:]
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=pad_idx
    )


def _greedy(model, features, vocab, max_steps=33):
    tokens = torch.full((features.shape[0], 1), vocab.start_idx, dtype=torch.long)
    for _ in range(max_steps):
        logits, _ = model(features, tokens)
        tokens = torch.cat((tokens, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
        if (tokens == vocab.end_idx).any(dim=1).all():
            break
    return [vocab.decode(row) for row in tokens]


def test_forward_shapes_and_validation():
    torch.manual_seed(42)
    vocab = _vocab()
    model = CaptionLSTM(len(vocab), padding_idx=vocab.pad_idx).eval()
    tokens = torch.randint(0, len(vocab), (3, 33))
    for features in (torch.randn(3, 2048, 7, 7), torch.randn(3, 49, 2048)):
        logits, attention = model(features, tokens)
        assert logits.shape == (3, 33, len(vocab)) and attention is None
    flat = torch.randn(2, 49, 2048)
    grid = flat.transpose(1, 2).reshape(2, 2048, 7, 7)
    torch.testing.assert_close(model(flat, tokens[:2, :5])[0], model(grid, tokens[:2, :5])[0])
    for bad_features, bad_tokens in (
        (torch.randn(3, 2048), tokens),
        (torch.randn(3, 49, 2048), tokens[:, :0]),
        (torch.randn(3, 49, 2048), torch.randint(0, len(vocab), (3, 34))),
        (torch.randn(2, 49, 2048), tokens),
    ):
        try:
            model(bad_features, bad_tokens)
        except ValueError:
            continue
        raise AssertionError("invalid input was accepted")


def test_half_precision_features():
    torch.manual_seed(42)
    vocab = _vocab()
    model = CaptionLSTM(len(vocab), padding_idx=vocab.pad_idx).eval()
    features = torch.randn(2, 49, 2048).half()
    tokens = torch.randint(0, len(vocab), (2, 7))
    logits, _ = model(features, tokens)
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    if torch.cuda.is_available():
        model = model.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(features.cuda(), tokens.cuda())
        assert torch.isfinite(logits).all()


def test_padding_is_ignored_by_loss():
    torch.manual_seed(42)
    vocab = _vocab()
    model = CaptionLSTM(len(vocab), padding_idx=vocab.pad_idx).eval()
    token_ids = torch.tensor([vocab.encode(c) for c in CAPTIONS[:2]])
    features = torch.randn(2, 49, 2048)
    logits, _ = model(features, token_ids[:, :-1])
    targets = token_ids[:, 1:]
    mask = targets != vocab.pad_idx
    expected = nn.functional.cross_entropy(logits[mask], targets[mask])
    torch.testing.assert_close(_loss(model, features, token_ids, vocab.pad_idx), expected)


def test_overfits_small_batch(steps=200, verbose=False):
    torch.manual_seed(42)
    vocab = _vocab()
    model = CaptionLSTM(len(vocab), padding_idx=vocab.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    token_ids = torch.tensor([vocab.encode(c) for c in CAPTIONS])
    features = torch.randn(len(CAPTIONS), 2048, 7, 7).relu()
    first_loss = None
    for step in range(steps):
        loss = _loss(model, features, token_ids, vocab.pad_idx)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        first_loss = loss.item() if first_loss is None else first_loss
        if verbose and step % 50 == 0:
            print(f"step {step:3d} loss {loss.item():.4f}")
    final_loss = loss.item()
    model.eval()
    with torch.no_grad():
        generated = _greedy(model, features, vocab)
    references = [vocab.decode(row) for row in token_ids]
    if verbose:
        print(f"loss {first_loss:.4f} -> {final_loss:.4f}")
        for reference, caption in zip(references, generated):
            print(f"  ref: {reference}\n  gen: {caption}")
    assert final_loss < 0.1 * first_loss
    assert generated == references


if __name__ == "__main__":
    vocab = _vocab()
    model = CaptionLSTM(len(vocab), padding_idx=vocab.pad_idx)
    print(f"vocab {len(vocab)}, parameters {sum(p.numel() for p in model.parameters()):,}")
    test_forward_shapes_and_validation()
    test_half_precision_features()
    test_padding_is_ignored_by_loss()
    test_overfits_small_batch(verbose=True)
    print("all checks passed")

import unittest
from pathlib import Path

import torch

from src.data.vocab import Vocabulary
from src.train import build_model, load_run_config


class TransformerTests(unittest.TestCase):
    def test_causality_spatial_attention_and_backward(self):
        root = Path(__file__).resolve().parents[1]
        vocab = Vocabulary.load(root / "src/data/artifacts/vocab.json")
        config = load_run_config(root / "configs/model3.yaml")
        torch.manual_seed(42)
        model = build_model(config, vocab).eval()
        features = torch.randn(2, 2048, 7, 7, dtype=torch.float16)
        ids = torch.tensor([[vocab.start_idx, 5, 6], [vocab.start_idx, 7, vocab.pad_idx]])
        logits, attention = model(features, ids)
        self.assertEqual(logits.shape, (2, 3, len(vocab)))
        self.assertEqual(attention.shape, (2, 3, 49))
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 3))
        changed = ids.clone()
        changed[:, 2] = 8
        other, _ = model(features.flatten(2).transpose(1, 2), changed)
        torch.testing.assert_close(logits[:, :2], other[:, :2])
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, len(vocab)),
                                                 ids[:, 1:].reshape(-1), ignore_index=vocab.pad_idx)
        loss.backward()
        self.assertGreater(model.projection.weight.grad.norm().item(), 0)

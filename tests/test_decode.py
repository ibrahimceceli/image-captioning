import unittest

import torch
from torch import nn

from src.data.vocab import Vocabulary
from src.decode import decode, generate_caption
from src.models.attention_lstm import AttentionLSTM


class FakeModel(nn.Module):
    def __init__(self, attention=True, end_after=2, fail=False):
        super().__init__()
        self.child = nn.Dropout()
        self.has_attention = attention
        self.end_after = end_after
        self.fail = fail
        self.steps = []

    def forward(self, features, tokens):
        if self.fail:
            raise RuntimeError("model failure")
        step = tokens.shape[1]
        self.steps.append(step)
        logits = torch.full((1, step, 6), -torch.inf)
        logits[0, -1, 2 if step > self.end_after else 4 + (step - 1) % 2] = 0
        weights = torch.zeros(1, step, 49) if self.has_attention else None
        if weights is not None:
            weights[0, -1, step % 49] = 1
        return logits, weights


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary(["<pad>", "<start>", "<end>", "<unk>", "a", "bird"], 5, 34)
        self.features = torch.zeros(1, 49, 2048)
        self.options = dict(start_id=self.vocab.start_idx, end_id=self.vocab.end_idx,
                            pad_id=self.vocab.pad_idx)

    def generate(self, model, **kwargs):
        return generate_caption(model, self.features, dict(enumerate(self.vocab.itos)),
                                **self.options, **kwargs)

    def test_methods_vocabulary_and_attention(self):
        for method in ("greedy", "beam", "nucleus"):
            for attention in (False, True):
                with self.subTest(method=method, attention=attention):
                    model = FakeModel(attention)
                    kwargs = dict(top_p=0.9, seed=42) if method == "nucleus" else {}
                    for vocab in (self.vocab, self.vocab.itos):
                        caption = decode(model, self.features, method, 3, vocab=vocab, **kwargs)
                        self.assertIsInstance(caption, str)
                        self.assertEqual(caption, "a bird")
                    caption, weights = self.generate(model, method=method, return_attention=True, **kwargs)
                    self.assertEqual(caption, "a bird")
                    if attention:
                        self.assertEqual(weights.shape, (2, 49))
                        self.assertEqual(weights.argmax(-1).tolist(), [1, 2])
                    else:
                        self.assertIsNone(weights)

    def test_boundaries_and_beam_one(self):
        for attention in (False, True):
            for end_after in (0, 2, 100):
                greedy = self.generate(FakeModel(attention, end_after), return_attention=True)
                beam = self.generate(FakeModel(attention, end_after), method="beam", beam_size=1,
                                     return_attention=True)
                self.assertEqual(greedy[0], beam[0])
                if attention:
                    torch.testing.assert_close(greedy[1], beam[1])
                for method in ("greedy", "beam", "nucleus"):
                    model = FakeModel(attention, end_after)
                    caption, weights = self.generate(model, method=method, top_p=0.9, seed=42,
                                                     return_attention=True)
                    self.assertEqual(len(caption.split()), min(end_after, 33))
                    self.assertLessEqual(max(model.steps), 33)
                    if attention:
                        self.assertEqual(weights.shape, (min(end_after, 33), 49))
                    else:
                        self.assertIsNone(weights)

    def test_modes_restored_even_on_failure(self):
        for training in (False, True):
            for fail in (False, True):
                for method in ("greedy", "beam", "nucleus"):
                    model = FakeModel(fail=fail)
                    model.train(training)
                    model.child.train(not training)
                    before = [module.training for module in model.modules()]
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "model failure"):
                            decode(model, self.features, method, 3, vocab=self.vocab, top_p=0.9)
                    else:
                        decode(model, self.features, method, 3, vocab=self.vocab, top_p=0.9)
                    self.assertEqual(before, [module.training for module in model.modules()])

    def test_invalid_vocabulary(self):
        for vocab in (None, [], ["bird"], ["<pad>", "<start>", "<end>", "bird", "bird"]):
            with self.subTest(vocab=vocab), self.assertRaises(ValueError):
                decode(FakeModel(), self.features, "greedy", 1, vocab=vocab)

    def test_special_ids_are_not_assumed(self):
        words = ["bird", "<end>", "<pad>", "<start>", "a", "<unk>"]
        # End immediately using the actual index in this reordered itos list.
        model = FakeModel(end_after=0)
        model.forward = lambda features, tokens: (torch.tensor([[[0., 10., 0., 0., 0., 0.]]]), None)
        self.assertEqual(decode(model, self.features, "greedy", 1, vocab=words), "")

    def test_real_model2_word_alignment(self):
        torch.manual_seed(42)
        model = AttentionLSTM(6, embedding_dim=4, hidden_dim=4, attention_dim=4, dropout=0)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.fill_(-100)
            model.output.bias[4] = 100
            tokens = torch.tensor([[self.vocab.start_idx, 4]])
            _, expected = model(self.features, tokens)
        caption, weights = self.generate(model, return_attention=True)
        self.assertEqual(caption.split(), ["a"] * 33)
        self.assertEqual(weights.shape, (33, 49))
        torch.testing.assert_close(weights[:2], expected[0])


if __name__ == "__main__":
    unittest.main()

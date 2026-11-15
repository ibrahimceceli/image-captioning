"""Exercise BLIP's real processor and decoder offline with a tiny synthetic checkpoint."""
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from transformers import (BertTokenizerFast, BlipConfig, BlipForConditionalGeneration,
                          BlipImageProcessor, BlipProcessor)

from src.blip import collate_training, generation_options, load_blip


def test_blip_backward_and_generation():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)
        (path / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n[DEC]\na\ndog\n")
        tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), bos_token="[DEC]")
        processor = BlipProcessor(BlipImageProcessor(size={"height": 32, "width": 32}), tokenizer)
        common = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4)
        config = BlipConfig(text_config=dict(common, vocab_size=len(tokenizer),
                                             bos_token_id=5, sep_token_id=3, pad_token_id=0),
                            vision_config=dict(common, image_size=32, patch_size=16))
        BlipForConditionalGeneration(config).save_pretrained(path)
        processor.save_pretrained(path)
        processor, model = load_blip(path, torch.device("cpu"), dropout=0.1)
        image = Image.new("RGB", (32, 32))
        batch = collate_training([(image, "a dog"), (image, "dog")], processor, 5)
        assert batch.input_ids[:, 0].tolist() == [5, 5]
        loss = model(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert model.vision_model.embeddings.patch_embedding.weight.grad is not None
        model.eval()
        for method in ("greedy", "beam"):
            tokens = model.generate(pixel_values=batch.pixel_values, **generation_options(method))
            assert tokens.shape[0] == 2

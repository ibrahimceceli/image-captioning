"""Reusable Model 2 demo adapter: construct once, call generate(PIL image).

Relative checkpoint/config paths are resolved against the repository root.
ResNet50 weights use torchvision's standard cache (TORCH_HOME is supported).
No feature cache, captions file or dataset split is needed for inference.
"""

from pathlib import Path

import torch
from PIL import Image

from src.data.config import load_config, repo_root
from src.data.features import build_transform, extract_image, load_encoder
from src.data.vocab import Vocabulary
from src.decode import decode
from src.models.attention_lstm import AttentionLSTM


class AttentionCaptioner:
    """Keep decoder, encoder and vocabulary loaded across image requests.

    Input is an unnormalized PIL.Image.Image of any size; it is converted
    to RGB, resized, center-cropped and ImageNet-normalized by the shared
    feature pipeline. For Gradio use an Image component with type="pil".
    The caller creates and retains this object outside the UI callback.
    Checkpoints are loaded with weights_only=True, so they cannot run code.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        config_path: str | Path | None = None,
        method: str = "greedy",
        beam_size: int = 3,
        top_p: float | None = None,
        seed: int = 42,
    ):
        if method not in ("greedy", "beam", "nucleus"):
            raise ValueError("method must be greedy, beam or nucleus")
        if method == "beam" and (type(beam_size) is not int or beam_size < 1):
            raise ValueError("beam_size must be a positive integer")
        if method == "nucleus" and (top_p is None or not 0 < top_p <= 1):
            raise ValueError("nucleus requires top_p in (0, 1]")
        root = repo_root()
        config = load_config(root / config_path if config_path is not None else None)
        self.device = torch.device(device)
        self.vocab = Vocabulary.load(root / config["vocab_path"])
        state = torch.load(root / checkpoint, map_location="cpu", weights_only=True)
        params = dict(state["config"]["model"])
        if params.pop("type") != "attention_lstm":
            raise ValueError("AttentionCaptioner requires an attention_lstm checkpoint")
        if state.get("vocab_size", len(self.vocab)) != len(self.vocab):
            raise ValueError("Checkpoint and vocabulary sizes do not match")
        self.model = AttentionLSTM(
            vocab_size=len(self.vocab), padding_idx=self.vocab.pad_idx, **params
        )
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.encoder = load_encoder(self.device)
        self.transform = build_transform(config)
        self.method, self.beam_size = method, beam_size
        self.top_p, self.seed = top_p, seed

    @torch.no_grad()
    def generate(self, image: Image.Image) -> str:
        """Extract spatial features from a new raw image and return its caption."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image (Gradio type='pil')")
        features = extract_image(image, self.encoder, self.transform, self.device)
        return decode(
            self.model, features, method=self.method, beam_size=self.beam_size,
            vocab=self.vocab, top_p=self.top_p, seed=self.seed,
        )

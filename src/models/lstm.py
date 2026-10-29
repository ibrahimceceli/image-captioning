"""Model 1: Show-and-Tell style CNN-LSTM baseline caption decoder."""

import torch
from torch import Tensor, nn


class CaptionLSTM(nn.Module):
    """Decode cached ResNet50 layer4 features using teacher forcing.

    Args:
        vocab_size: Size of the shared vocabulary, including special tokens.
        embedding_dim: Size of the word embeddings.
        hidden_dim: Size of the LSTM hidden and cell states.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout before the output layer (and between LSTM layers).
        padding_idx: Shared padding token ID, if known; no ID is assumed.

    ``forward(features, caption_inputs)`` accepts features shaped
    [B, 2048, 7, 7] or [B, 49, 2048] and integer token IDs [B, T].
    It returns raw logits [B, T, vocab_size] and ``None`` in place of
    attention, because this baseline sees a single global image vector.
    Each input token predicts the next token: the caller supplies captions
    without their final token as inputs and captions without their first
    token as targets. A 34-token caption therefore needs at most 33 steps.
    Padding masks, loss calculation, and sequence generation belong to callers.
    Features whose dtype differs from the weights (e.g. a float16 cache)
    are cast to the weight dtype, so autocast and float32 runs both work.
    Set torch.manual_seed(42) before construction for reproducible initialization.
    """

    feature_dim = 2048
    num_regions = 49
    max_caption_length = 34

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.5,
        padding_idx: int | None = None,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.initial_hidden = nn.Linear(self.feature_dim, hidden_dim)
        self.initial_cell = nn.Linear(self.feature_dim, hidden_dim)
        # nn.LSTM applies dropout only between layers; a single layer needs none.
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, features: Tensor, caption_inputs: Tensor
    ) -> tuple[Tensor, None]:
        if features.ndim == 4 and tuple(features.shape[1:]) == (2048, 7, 7):
            regions = features.flatten(2).transpose(1, 2)
        elif features.ndim == 3 and tuple(features.shape[1:]) == (49, 2048):
            regions = features
        else:
            raise ValueError("features must have shape [B, 2048, 7, 7] or [B, 49, 2048]")
        if caption_inputs.ndim != 2 or caption_inputs.shape[0] != regions.shape[0]:
            raise ValueError("caption_inputs must have shape [B, T] matching features")
        if not 1 <= caption_inputs.shape[1] <= self.max_caption_length - 1:
            raise ValueError("caption_inputs must contain 1 to 33 next-token prediction steps")

        weight_dtype = self.initial_hidden.weight.dtype
        if regions.dtype != weight_dtype:
            regions = regions.to(weight_dtype)
        # The image is seen only once, as the initial recurrent state built from
        # the mean of the 49 regions (global average pooling).
        mean_features = regions.mean(dim=1)
        hidden = torch.tanh(self.initial_hidden(mean_features))
        cell = torch.tanh(self.initial_cell(mean_features))
        hidden = hidden.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        cell = cell.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        outputs, _ = self.lstm(self.embedding(caption_inputs), (hidden, cell))
        return self.output(self.dropout(outputs)), None

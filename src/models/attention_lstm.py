"""Model 2: region-based Bahdanau attention and an LSTM caption decoder."""

import torch
from torch import Tensor, nn


class BahdanauAttention(nn.Module):
    """Compute a context vector and normalized weights over image regions."""

    def __init__(self, feature_dim: int, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, attention_dim)
        self.hidden_projection = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, regions: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        energy = self.score(torch.tanh(
            self.feature_projection(regions)
            + self.hidden_projection(hidden).unsqueeze(1)
        )).squeeze(-1)
        weights = torch.softmax(energy, dim=1)
        context = torch.sum(regions * weights.unsqueeze(-1), dim=1)
        return context, weights


class AttentionLSTM(nn.Module):
    """Decode cached ResNet50 layer4 features using teacher forcing.

    Args:
        vocab_size: Size of the shared vocabulary, including special tokens.
        padding_idx: Shared padding token ID, if known; no ID is assumed.

    ``forward(features, caption_inputs)`` accepts features shaped
    [B, 2048, 7, 7] or [B, 49, 2048] and integer token IDs [B, T].
    It returns raw logits [B, T, vocab_size] and attention [B, T, 49].
    Each input token predicts the next token: the caller supplies captions
    without their final token as inputs and captions without their first
    token as targets. A 34-token caption therefore needs at most 33 steps.
    Padding masks, loss calculation, and sequence generation belong to callers.
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
        attention_dim: int = 512,
        dropout: float = 0.5,
        padding_idx: int | None = None,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.attention = BahdanauAttention(
            self.feature_dim, hidden_dim, attention_dim
        )
        self.lstm = nn.LSTMCell(embedding_dim + self.feature_dim, hidden_dim)
        self.initial_hidden = nn.Linear(self.feature_dim, hidden_dim)
        self.initial_cell = nn.Linear(self.feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, features: Tensor, caption_inputs: Tensor
    ) -> tuple[Tensor, Tensor]:
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

        # The mean initializes the recurrent state only. Attention still receives
        # all 49 separate regions at every step.
        mean_features = regions.mean(dim=1)
        hidden = torch.tanh(self.initial_hidden(mean_features))
        cell = torch.tanh(self.initial_cell(mean_features))
        embeddings = self.embedding(caption_inputs)
        logits, attention_weights = [], []
        for embedding in embeddings.unbind(dim=1):
            context, weights = self.attention(regions, hidden)
            hidden, cell = self.lstm(torch.cat((embedding, context), dim=1), (hidden, cell))
            logits.append(self.output(self.dropout(hidden)))
            attention_weights.append(weights)
        return torch.stack(logits, dim=1), torch.stack(attention_weights, dim=1)

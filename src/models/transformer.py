"""Causal caption decoder over the 7 × 7 ResNet50 feature grid."""
import torch
from torch import nn


class DecoderBlock(nn.Module):
    def __init__(self, width, heads, ffn_dim, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(width, heads, dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, width))
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, causal, padding, return_weights):
        update, _ = self.self_attention(x, x, x, attn_mask=causal,
                                        key_padding_mask=padding, need_weights=False)
        x = self.norm1(x + self.dropout(update))
        update, weights = self.cross_attention(x, memory, memory, need_weights=return_weights)
        x = self.norm2(x + self.dropout(update))
        return self.norm3(x + self.dropout(self.ffn(x))), weights


class CaptionTransformer(nn.Module):
    def __init__(self, vocab_size, padding_idx, d_model=256, num_heads=8,
                 num_layers=3, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.word_position = nn.Embedding(34, d_model)
        self.projection = nn.Linear(2048, d_model)
        self.row = nn.Embedding(7, d_model)
        self.column = nn.Embedding(7, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(DecoderBlock(d_model, num_heads, ffn_dim, dropout)
                                    for _ in range(num_layers))
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, features, caption_inputs):
        if features.shape[1:] == (2048, 7, 7):
            features = features.flatten(2).transpose(1, 2)
        if features.shape[1:] != (49, 2048):
            raise ValueError("Expected 7 x 7 ResNet50 features")
        indices = torch.arange(caption_inputs.size(1), device=caption_inputs.device)
        causal = indices[:, None] < indices[None, :]
        padding = caption_inputs.eq(self.embedding.padding_idx)
        spatial = self.row.weight[:, None, :] + self.column.weight[None, :, :]
        memory = self.projection(features.to(self.projection.weight.dtype)) + spatial.flatten(0, 1)
        x = self.dropout(self.embedding(caption_inputs) + self.word_position(indices))
        for index, block in enumerate(self.blocks):
            x, weights = block(x, memory, causal, padding, index == len(self.blocks) - 1)
        return self.output(x), weights

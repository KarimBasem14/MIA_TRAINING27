"""Attention-based image captioning model."""

import torch
from torch import nn


class CaptionModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, vocab_size: int, pad_id: int, dropout: float = 0.3):
        super().__init__()
        self.pad_id = pad_id
        self.feature_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.lstm = nn.LSTMCell(hidden_dim * 2, hidden_dim)
        self.attention = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, vocab_size))

    def forward(self, features: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced training forward pass. Features are (batch, regions, feature_dim)."""
        projected = self.feature_projection(features)
        batch_size = features.size(0)
        hidden = projected.mean(dim=1)
        cell = torch.zeros_like(hidden)
        logits = []
        for step in range(caption_ids.size(1) - 1):
            query = hidden.unsqueeze(1).expand_as(projected)
            scores = self.attention(torch.cat([projected, query], dim=-1)).tanh().sum(dim=-1)
            weights = scores.softmax(dim=1)
            context = (weights.unsqueeze(-1) * projected).sum(dim=1)
            decoder_input = torch.cat([self.embedding(caption_ids[:, step]), context], dim=-1)
            hidden, cell = self.lstm(decoder_input, (hidden, cell))
            logits.append(self.output(hidden))
        return torch.stack(logits, dim=1) if logits else torch.empty(batch_size, 0, self.output[-1].out_features)

    @torch.no_grad()
    def generate(self, features: torch.Tensor, start_id: int, end_id: int, max_length: int = 30) -> list[list[int]]:
        self.eval()
        projected = self.feature_projection(features)
        hidden = projected.mean(dim=1)
        cell = torch.zeros_like(hidden)
        tokens = torch.full((features.size(0), 1), start_id, dtype=torch.long, device=features.device)
        finished = torch.zeros(features.size(0), dtype=torch.bool, device=features.device)
        for _ in range(max_length):
            query = hidden.unsqueeze(1).expand_as(projected)
            scores = self.attention(torch.cat([projected, query], dim=-1)).tanh().sum(dim=-1)
            context = (scores.softmax(dim=1).unsqueeze(-1) * projected).sum(dim=1)
            hidden, cell = self.lstm(torch.cat([self.embedding(tokens[:, -1]), context], dim=-1), (hidden, cell))
            next_token = self.output(hidden).argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, end_id), next_token)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(end_id)
            if finished.all():
                break
        return tokens.cpu().tolist()

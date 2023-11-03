from __future__ import annotations

import torch


def overlapped_speech_penalty(
    segmentation: torch.Tensor, gamma: float = 3, beta: float = 10
):
    # segmentation has shape (batch, frames, speakers)
    probs = torch.softmax(beta * segmentation, dim=-1)
    weights = torch.pow(segmentation, gamma) * torch.pow(probs, gamma)
    weights[weights < 1e-8] = 1e-8
    return weights


def normalize_embeddings(
    embeddings: torch.Tensor, norm: float | torch.Tensor = 1
) -> torch.Tensor:
    # embeddings has shape (batch, speakers, feat) or (speakers, feat)
    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(0)
    if isinstance(norm, torch.Tensor):
        batch_size1, num_speakers1, _ = norm.shape
        batch_size2, num_speakers2, _ = embeddings.shape
        assert batch_size1 == batch_size2 and num_speakers1 == num_speakers2
    emb_norm = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
    return norm * embeddings / emb_norm

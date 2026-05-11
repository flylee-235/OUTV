from typing import Optional

import torch
import torch.nn as nn


class SemanticLayoutPredictor(nn.Module):
    """Small per-frame semantic completion head for SAM3 class masks."""

    def __init__(self, num_classes: int, hidden_dim: int = 64, relative_dim: int = 6):
        super().__init__()
        self.relative_mlp = nn.Sequential(
            nn.Linear(relative_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Conv2d(num_classes + 1 + hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )

    def forward(
        self,
        known_masks: torch.Tensor,
        unknown_mask: torch.Tensor,
        relative_position: torch.Tensor,
    ) -> torch.Tensor:
        if known_masks.ndim != 5:
            raise ValueError(f"known_masks must be B,F,K,H,W, got {tuple(known_masks.shape)}")
        b, f, _, h, w = known_masks.shape
        if unknown_mask.ndim == 4:
            unknown_mask = unknown_mask.unsqueeze(2)
        rel = self.relative_mlp(relative_position.float()).view(b, 1, -1, 1, 1)
        rel = rel.expand(b, f, -1, h, w)
        x = torch.cat([known_masks.float(), unknown_mask.float(), rel], dim=2)
        logits = self.net(x.reshape(b * f, x.shape[2], h, w))
        return logits.view(b, f, -1, h, w)


def semantic_layout_loss(
    logits: torch.Tensor,
    target_masks: torch.Tensor,
    unknown_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits.float(), target_masks.float(), reduction="none"
    )
    if unknown_mask is not None:
        if unknown_mask.ndim == 4:
            unknown_mask = unknown_mask.unsqueeze(2)
        loss = loss * unknown_mask.float()
        denom = unknown_mask.float().sum().clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()


"""Batch collation utilities."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("collate_fn received an empty batch")
    out: Dict[str, Any] = {"frames": torch.stack([b["frames"] for b in batch], dim=0)}
    if "input_ids" in batch[0]:
        out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
    if "attention_mask" in batch[0]:
        out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch], dim=0)
    if "prompt" in batch[0]:
        out["prompt"] = [b["prompt"] for b in batch]
    return out


def random_prompt_dropout(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          pad_id: int, prob: float,
                          generator: torch.Generator | None = None) -> torch.Tensor:
    if prob <= 0.0:
        return torch.zeros(input_ids.shape[0], device=input_ids.device)
    if generator is not None:
        rand = torch.rand(input_ids.shape[0], generator=generator)
    else:
        rand = torch.rand(input_ids.shape[0])
    return (rand < prob).float().to(input_ids.device)

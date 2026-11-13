from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Optional

import torch

from .types import MODALITIES, PERBatch, PEROutput


def _cat(values: Iterable[torch.Tensor]) -> Optional[torch.Tensor]:
    items = [value.detach().float().cpu().reshape(-1) for value in values]
    items = [value for value in items if value.numel() > 0]
    return torch.cat(items) if items else None


def _stats(values: Optional[torch.Tensor], prefix: str) -> Dict[str, float]:
    if values is None or values.numel() == 0:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p05": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
        }
    return {
        f"{prefix}_count": float(values.numel()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(unbiased=False)),
        f"{prefix}_p05": float(torch.quantile(values, 0.05)),
        f"{prefix}_p50": float(torch.quantile(values, 0.50)),
        f"{prefix}_p95": float(torch.quantile(values, 0.95)),
    }


class RouteDiagnosticsAccumulator:
    """Collect detached route statistics over active missing positions."""

    def __init__(self) -> None:
        self.route = []
        self.route_by_modality: Dict[str, List[torch.Tensor]] = {
            modality: [] for modality in MODALITIES
        }
        self.route_by_pattern: Dict[str, List[torch.Tensor]] = {}
        self.route_target = []
        self.utility_gap = []

    def add(
        self,
        output: PEROutput,
        batch: PERBatch,
        route_target: Optional[torch.Tensor] = None,
        utility_gap: Optional[torch.Tensor] = None,
    ) -> None:
        active = (~batch.availability) & batch.sequence_mask.unsqueeze(-1)
        route = output.route.detach()
        if active.any():
            self.route.append(route[active])
        for modality_index, modality in enumerate(MODALITIES):
            selected = active[:, :, modality_index]
            if selected.any():
                self.route_by_modality[modality].append(
                    route[:, :, modality_index][selected]
                )

        available = batch.availability.detach().cpu()
        sequence = batch.sequence_mask.detach().cpu()
        route_cpu = route.cpu()
        for pattern in product((False, True), repeat=3):
            if not any(pattern):
                continue
            key = "".join("1" if value else "0" for value in pattern)
            selected = (
                ((~available) == torch.tensor(pattern, dtype=torch.bool))
                & sequence.unsqueeze(-1)
                & (~available)
            )
            if selected.any():
                self.route_by_pattern.setdefault(key, []).append(route_cpu[selected])

        if route_target is not None and active.any():
            self.route_target.append(route_target.detach()[active])
        if utility_gap is not None and active.any():
            self.utility_gap.append(utility_gap.detach()[active])

    def summary(self) -> Dict[str, float]:
        route = _cat(self.route)
        target = _cat(self.route_target)
        gap = _cat(self.utility_gap)
        result = _stats(route, "route")
        if route is not None and route.numel():
            result["route_low_fraction"] = float((route < 0.05).float().mean())
            result["route_high_fraction"] = float((route > 0.95).float().mean())
        else:
            result["route_low_fraction"] = 0.0
            result["route_high_fraction"] = 0.0

        for modality in MODALITIES:
            values = _cat(self.route_by_modality[modality])
            result[f"route_{modality}_mean"] = (
                float(values.mean()) if values is not None else 0.0
            )
        for pattern, chunks in self.route_by_pattern.items():
            values = _cat(chunks)
            result[f"route_pattern_{pattern}_mean"] = (
                float(values.mean()) if values is not None else 0.0
            )

        result.update(_stats(target, "route_target"))
        result.update(_stats(gap, "teacher_utility_gap"))
        correlation = 0.0
        if (
            route is not None
            and target is not None
            and route.numel() == target.numel()
            and route.numel() > 1
            and route.std(unbiased=False) > 1e-8
            and target.std(unbiased=False) > 1e-8
        ):
            value = torch.corrcoef(torch.stack([route, target]))[0, 1]
            if torch.isfinite(value):
                correlation = float(value)
        result["route_target_correlation"] = correlation
        return result

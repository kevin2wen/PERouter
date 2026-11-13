from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch

from .teacher_diagnostics import pearson, spearman


def _entropy(weights: torch.Tensor) -> torch.Tensor:
    probabilities = weights.clamp_min(1e-8)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(probabilities * probabilities.log()).sum(dim=-1)


class FusionDiagnosticsAccumulator:
    def __init__(self) -> None:
        self.route: List[torch.Tensor] = []
        self.coordinator: List[torch.Tensor] = []
        self.entropy: List[torch.Tensor] = []
        self.patterns: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(
            lambda: {"route": [], "coordinator": []}
        )

    def add(self, output, batch) -> None:
        if output.coordinator_weights is None:
            return
        active = (
            (~batch.availability)
            & batch.sequence_mask.unsqueeze(-1)
        )
        if batch.evidence is not None:
            evidence_available = (
                batch.evidence.mask.any(dim=-1).unsqueeze(-1).expand(-1, -1, 3)
            )
            active = active & evidence_available
        if not active.any():
            return
        route = output.route.detach()
        coordinator = output.coordinator_weights.detach()
        self.route.append(route[active])
        self.coordinator.append(coordinator[active])
        self.entropy.append(_entropy(coordinator)[active.any(dim=-1)])
        missing = (~batch.availability).detach().cpu()
        active_cpu = active.detach().cpu()
        route_cpu = route.cpu()
        coordinator_cpu = coordinator.cpu()
        for bits in torch.cartesian_prod(
            torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 1])
        ):
            if not bits.any():
                continue
            key = "".join(str(int(value)) for value in bits.tolist())
            selected = (missing == bits.bool().view(1, 1, 3)) & active_cpu
            if selected.any():
                self.patterns[key]["route"].append(route_cpu[selected])
                self.patterns[key]["coordinator"].append(coordinator_cpu[selected])

    @staticmethod
    def _cat(values):
        values = [value.reshape(-1).float() for value in values if value.numel()]
        return torch.cat(values) if values else None

    def summary(self) -> Dict[str, float]:
        route = self._cat(self.route)
        coordinator = self._cat(self.coordinator)
        entropy = self._cat(self.entropy)
        result: Dict[str, float] = {
            "fusion_count": float(route.numel()) if route is not None else 0.0,
            "fusion_route_coordinator_pearson": 0.0,
            "fusion_route_coordinator_spearman": 0.0,
            "fusion_coordinator_entropy_mean": 0.0,
        }
        if route is not None and coordinator is not None:
            result["fusion_route_coordinator_pearson"] = pearson(route, coordinator)
            result["fusion_route_coordinator_spearman"] = spearman(route, coordinator)
        if entropy is not None and entropy.numel():
            result["fusion_coordinator_entropy_mean"] = float(entropy.mean())
        for pattern, values in self.patterns.items():
            pattern_route = self._cat(values["route"])
            pattern_coordinator = self._cat(values["coordinator"])
            result[f"fusion_pattern_{pattern}_count"] = (
                float(pattern_route.numel()) if pattern_route is not None else 0.0
            )
            result[f"fusion_pattern_{pattern}_route_coordinator_pearson"] = (
                pearson(pattern_route, pattern_coordinator)
                if pattern_route is not None and pattern_coordinator is not None
                else 0.0
            )
        return result

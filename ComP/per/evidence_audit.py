from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch

from .types import MODALITIES, EvidenceBatch


def expanded_evidence_available(evidence: EvidenceBatch) -> torch.BoolTensor:
    if evidence.mask.ndim != 3:
        raise ValueError("evidence mask must be [B,T,K]")
    return evidence.mask.any(dim=-1).unsqueeze(-1).expand(-1, -1, 3)


def compute_evidence_coverage(
    evidence: EvidenceBatch,
    availability: torch.BoolTensor,
    sequence_mask: torch.BoolTensor,
) -> Dict[str, float]:
    if availability.shape != (*sequence_mask.shape, 3):
        raise ValueError("availability must be [B,T,3]")
    available = expanded_evidence_available(evidence)
    if available.shape != availability.shape:
        raise ValueError("expanded evidence availability shape mismatch")
    missing = (~availability) & sequence_mask.unsqueeze(-1)
    covered = missing & available
    result: Dict[str, float] = {
        "coverage_all_numerator": float(covered.sum().item()),
        "coverage_all_denominator": float(missing.sum().item()),
        "coverage_all_fraction": (
            float(covered.sum().item()) / max(1, int(missing.sum().item()))
        ),
    }
    for modality_index, modality in enumerate(MODALITIES):
        denominator = int(missing[:, :, modality_index].sum().item())
        numerator = int(covered[:, :, modality_index].sum().item())
        result[f"coverage_{modality}_numerator"] = float(numerator)
        result[f"coverage_{modality}_denominator"] = float(denominator)
        result[f"coverage_{modality}_fraction"] = (
            numerator / denominator if denominator else 0.0
        )
    missing_pattern = (~availability).detach().cpu()
    sequence_cpu = sequence_mask.detach().cpu()
    covered_cpu = covered.detach().cpu()
    for bits in torch.cartesian_prod(
        torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 1])
    ):
        key = "".join(str(int(value)) for value in bits.tolist())
        if key == "000":
            continue
        pattern = bits.bool().view(1, 1, 3)
        denominator = int(
            ((missing_pattern == pattern) & sequence_cpu.unsqueeze(-1)).sum().item()
        )
        numerator = int(
            ((missing_pattern == pattern) & covered_cpu).sum().item()
        )
        result[f"coverage_pattern_{key}_numerator"] = float(numerator)
        result[f"coverage_pattern_{key}_denominator"] = float(denominator)
        result[f"coverage_pattern_{key}_fraction"] = (
            numerator / denominator if denominator else 0.0
        )
    return result


class EvidenceCoverageAccumulator:
    def __init__(self) -> None:
        self.counts: Dict[str, int] = defaultdict(int)

    def add(
        self,
        evidence: EvidenceBatch,
        availability: torch.BoolTensor,
        sequence_mask: torch.BoolTensor,
    ) -> None:
        values = compute_evidence_coverage(evidence, availability, sequence_mask)
        for key, value in values.items():
            if key.endswith("numerator") or key.endswith("denominator"):
                self.counts[key] += int(value)

    def summary(self) -> Dict[str, float]:
        result = {}
        all_num = self.counts.get("coverage_all_numerator", 0)
        all_den = self.counts.get("coverage_all_denominator", 0)
        result["coverage_all_numerator"] = float(all_num)
        result["coverage_all_denominator"] = float(all_den)
        result["coverage_all_fraction"] = all_num / all_den if all_den else 0.0
        prefixes = [f"coverage_{name}" for name in MODALITIES]
        prefixes.extend(
            f"coverage_pattern_{key}"
            for key in (
                "001", "010", "011", "100", "101", "110", "111"
            )
        )
        for prefix in prefixes:
            numerator = self.counts.get(f"{prefix}_numerator", 0)
            denominator = self.counts.get(f"{prefix}_denominator", 0)
            result[f"{prefix}_numerator"] = float(numerator)
            result[f"{prefix}_denominator"] = float(denominator)
            result[f"{prefix}_fraction"] = (
                numerator / denominator if denominator else 0.0
            )
        return result


def evidence_trace_rows(
    evidence: EvidenceBatch,
    availability: torch.BoolTensor,
    sequence_mask: torch.BoolTensor,
    *,
    split: str,
    sample_ids,
    utterance_ids,
    evidence_mode: str,
    mask_rate: float,
    seed: int,
    sampler_version: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    available = expanded_evidence_available(evidence).detach().cpu()
    availability_cpu = availability.detach().cpu()
    sequence_cpu = sequence_mask.detach().cpu()
    confidence = evidence.source_confidence.detach().cpu()
    for batch_index, sample_id in enumerate(sample_ids):
        for time_index, utterance_id in enumerate(utterance_ids[batch_index]):
            if not bool(sequence_cpu[batch_index, time_index]):
                continue
            for modality_index, modality in enumerate(MODALITIES):
                if bool(availability_cpu[batch_index, time_index, modality_index]):
                    continue
                availability_key = "".join(
                    "1" if bool(value) else "0"
                    for value in availability_cpu[batch_index, time_index].tolist()
                )
                for rank in range(evidence.mask.size(2)):
                    neighbor_id = evidence.ids[batch_index][time_index][rank]
                    if neighbor_id:
                        neighbor_sample, separator, neighbor_utterance = neighbor_id.partition(":")
                    else:
                        neighbor_sample, neighbor_utterance = "", ""
                        separator = ""
                    rows.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "utterance_id": utterance_id,
                            "availability": availability_key,
                            "target_modality": modality,
                            "evidence_rank": rank,
                            "neighbor_sample_id": neighbor_sample,
                            "neighbor_utterance_id": neighbor_utterance,
                            "source_confidence": float(
                                confidence[batch_index, time_index, rank]
                            ),
                            "evidence_available": bool(
                                available[batch_index, time_index, modality_index]
                            ),
                            "rank_available": bool(
                                evidence.mask[batch_index, time_index, rank]
                            ),
                            "evidence_mode": evidence_mode,
                            "mask_rate": mask_rate,
                            "seed": seed,
                            "sampler_version": sampler_version,
                            "shared_across_target_modalities": True,
                        }
                    )
    return rows

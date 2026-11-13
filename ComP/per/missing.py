from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


VALID_PATTERNS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)


def _record_seed(seed: int, epoch: int, sample_id: str, stream: str) -> int:
    payload = f"{seed}|{epoch}|{sample_id}|{stream}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


def _validate_inputs(
    sample_ids: Sequence[str],
    sequence_mask: torch.BoolTensor,
) -> None:
    if sequence_mask.dtype != torch.bool:
        raise TypeError("sequence_mask must be torch.bool")
    if sequence_mask.ndim != 2:
        raise ValueError("sequence_mask must be [B,T]")
    if len(sample_ids) != sequence_mask.shape[0]:
        raise ValueError(
            f"sample_ids length={len(sample_ids)} but B={sequence_mask.shape[0]}"
        )


class MissingPatternSampler:
    """Deterministic per-utterance missing-pattern sampler.

    Sampling is keyed by seed, epoch, sample_id and time index, so moving a
    sample to another batch does not change its pattern. Invalid padding is
    always marked unavailable.
    """

    def __init__(
        self,
        allowed_patterns: Tuple[Tuple[int, int, int], ...] = VALID_PATTERNS,
        pattern_probabilities: Optional[torch.Tensor] = None,
        seed: int = 666,
        missing_rate: float = 0.7,
        protocol: str = "categorical",
    ):
        if not allowed_patterns:
            raise ValueError("allowed_patterns must not be empty")
        for pattern in allowed_patterns:
            if len(pattern) != 3 or any(value not in (0, 1) for value in pattern):
                raise ValueError(f"invalid three-modality pattern: {pattern}")
            if sum(pattern) == 0:
                raise ValueError("all-missing patterns are unsupported")
        if protocol not in {"categorical", "bernoulli", "legacy", "fixed"}:
            raise ValueError(f"unknown missing protocol: {protocol}")
        if not 0.0 <= missing_rate <= 1.0:
            raise ValueError("missing_rate must be in [0,1]")

        self.allowed_patterns = tuple(tuple(int(v) for v in p) for p in allowed_patterns)
        self.seed = int(seed)
        self.missing_rate = float(missing_rate)
        self.protocol = protocol
        if pattern_probabilities is None:
            probabilities = torch.full(
                (len(self.allowed_patterns),), 1.0 / len(self.allowed_patterns)
            )
        else:
            probabilities = torch.as_tensor(pattern_probabilities, dtype=torch.float32)
            if probabilities.shape != (len(self.allowed_patterns),):
                raise ValueError(
                    "pattern_probabilities must have one value per allowed pattern"
                )
            if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
                raise ValueError("pattern_probabilities must be finite and non-negative")
            total = probabilities.sum()
            if total <= 0:
                raise ValueError("pattern_probabilities must have positive sum")
            probabilities = probabilities / total
        self.pattern_probabilities = probabilities.cpu()
        self._uses_explicit_probabilities = pattern_probabilities is not None

    def _empty(self, sequence_mask: torch.BoolTensor) -> torch.BoolTensor:
        return torch.zeros(
            (*sequence_mask.shape, 3), dtype=torch.bool, device=sequence_mask.device
        )

    def _sample_categorical(
        self,
        sample_ids: Sequence[str],
        sequence_mask: torch.BoolTensor,
        epoch: int,
    ) -> torch.BoolTensor:
        availability = self._empty(sequence_mask)

        # Explicit probabilities retain the original seven-pattern behavior.
        # The default path maps mask_rate to an expected missing-modality count.
        if self._uses_explicit_probabilities:
            probabilities = self.pattern_probabilities
            patterns = torch.tensor(self.allowed_patterns, dtype=torch.bool)
            for batch_index, sample_id in enumerate(sample_ids):
                valid = sequence_mask[batch_index]
                n_valid = int(valid.sum().item())
                if n_valid == 0:
                    continue
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    _record_seed(self.seed, epoch, sample_id, "categorical")
                )
                choices = torch.multinomial(
                    probabilities,
                    num_samples=n_valid,
                    replacement=True,
                    generator=generator,
                )
                availability[batch_index, valid] = patterns[choices].to(
                    availability.device
                )
            return availability

        patterns_by_missing_count = {}
        for pattern in self.allowed_patterns:
            missing_count = 3 - sum(pattern)
            patterns_by_missing_count.setdefault(missing_count, []).append(pattern)

        available_counts = sorted(patterns_by_missing_count)
        if not available_counts:
            raise ValueError("allowed_patterns has no usable pattern")

        # There are at most two missing modalities because all-missing rows are
        # forbidden. Average missing fraction therefore saturates at 2/3.
        target_missing_count = min(2.0, max(0.0, 3.0 * self.missing_rate))
        lower_target = int(target_missing_count)
        upper_target = int(target_missing_count) + 1
        if upper_target > 2:
            upper_target = lower_target
        fraction_to_upper = target_missing_count - lower_target

        lower_count = min(
            available_counts,
            key=lambda count: (abs(count - lower_target), count),
        )
        upper_count = min(
            available_counts,
            key=lambda count: (abs(count - upper_target), count),
        )

        for batch_index, sample_id in enumerate(sample_ids):
            valid = sequence_mask[batch_index]
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _record_seed(self.seed, epoch, sample_id, "categorical")
            )

            if lower_count == upper_count or fraction_to_upper <= 0:
                missing_counts = torch.full(
                    (n_valid,), lower_count, dtype=torch.int64
                )
            else:
                choose_upper = torch.rand(
                    (n_valid,), generator=generator
                ) < fraction_to_upper
                missing_counts = torch.where(
                    choose_upper,
                    torch.full((n_valid,), upper_count, dtype=torch.int64),
                    torch.full((n_valid,), lower_count, dtype=torch.int64),
                )

            for missing_count in torch.unique(missing_counts).tolist():
                row_indices = torch.nonzero(
                    missing_counts == missing_count, as_tuple=False
                ).flatten()
                candidates = patterns_by_missing_count[int(missing_count)]
                candidate_tensor = torch.tensor(candidates, dtype=torch.bool)
                choices = torch.randint(
                    0,
                    len(candidates),
                    (row_indices.numel(),),
                    generator=generator,
                )
                selected_valid = torch.nonzero(valid, as_tuple=False).flatten()
                selected_rows = selected_valid[row_indices]
                availability[batch_index, selected_rows] = candidate_tensor[
                    choices
                ].to(availability.device)

        return availability

    def _sample_bernoulli(
        self,
        sample_ids: Sequence[str],
        sequence_mask: torch.BoolTensor,
        epoch: int,
    ) -> torch.BoolTensor:
        availability = self._empty(sequence_mask)
        for batch_index, sample_id in enumerate(sample_ids):
            valid = sequence_mask[batch_index]
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_record_seed(self.seed, epoch, sample_id, "bernoulli"))
            observed = torch.rand(
                (n_valid, 3), generator=generator, dtype=torch.float32
            ) >= self.missing_rate
            all_missing = ~observed.any(dim=-1)
            if all_missing.any():
                fallback = torch.randint(
                    0, 3, (int(all_missing.sum().item()),), generator=generator
                )
                observed[all_missing] = False
                observed[all_missing, fallback] = True
            availability[batch_index, valid] = observed.to(availability.device)
        return availability

    def _sample_legacy(
        self,
        sample_ids: Sequence[str],
        sequence_mask: torch.BoolTensor,
        epoch: int,
    ) -> torch.BoolTensor:
        # Local import avoids a cycle: train_mr imports modules that may import PER.
        comp_dir = str(Path(__file__).resolve().parents[1])
        inserted = comp_dir not in sys.path
        if inserted:
            sys.path.insert(0, comp_dir)
        try:
            from train_mr import random_mask
        finally:
            if inserted:
                sys.path.remove(comp_dir)

        batch_size, seq_len = sequence_mask.shape
        flat = random_mask(3, seq_len * batch_size, self.missing_rate)
        matrix = torch.as_tensor(flat, dtype=torch.bool).reshape(
            batch_size, seq_len, 3
        )
        return matrix.to(sequence_mask.device) & sequence_mask.unsqueeze(-1)

    def sample(
        self,
        sample_ids: List[str],
        sequence_mask: torch.BoolTensor,
        epoch: int,
    ) -> torch.BoolTensor:
        _validate_inputs(sample_ids, sequence_mask)
        epoch = int(epoch)
        if self.protocol == "categorical":
            return self._sample_categorical(sample_ids, sequence_mask, epoch)
        if self.protocol == "bernoulli":
            return self._sample_bernoulli(sample_ids, sequence_mask, epoch)
        if self.protocol == "legacy":
            return self._sample_legacy(sample_ids, sequence_mask, epoch)
        raise ValueError("fixed sampling requires sampler.fixed(...)")

    def fixed(
        self,
        sample_ids: List[str],
        pattern: Tuple[int, int, int],
        sequence_mask: torch.BoolTensor,
    ) -> torch.BoolTensor:
        _validate_inputs(sample_ids, sequence_mask)
        if len(pattern) != 3 or any(value not in (0, 1) for value in pattern):
            raise ValueError("pattern must be a three-element binary tuple")
        if sum(pattern) == 0:
            raise ValueError("all-missing pattern is unsupported")
        availability = self._empty(sequence_mask)
        availability[sequence_mask] = torch.tensor(
            pattern, dtype=torch.bool, device=availability.device
        )
        return availability


class MaskManifest:
    """CSV-backed fixed mask manifest.

    The public CSV fields include the required
    (seed, sample_id, utterance_index, availability) tuple.
    """

    FIELDS = (
        "seed",
        "sample_id",
        "utterance_index",
        "utterance_id",
        "availability",
        "mask_rate",
        "sampler_version",
    )

    @staticmethod
    def save(
        path: str,
        sample_ids: Sequence[str],
        utterance_ids: Sequence[Sequence[str]],
        availability: torch.BoolTensor,
        sequence_mask: torch.BoolTensor,
        *,
        seed: int,
        mask_rate: float,
        sampler_version: str,
    ) -> None:
        _validate_inputs(sample_ids, sequence_mask)
        if availability.shape != (*sequence_mask.shape, 3):
            raise ValueError("availability must match sequence_mask [B,T,3]")
        if availability.dtype != torch.bool:
            raise TypeError("availability must be torch.bool")
        if len(utterance_ids) != len(sample_ids):
            raise ValueError("utterance_ids must have shape [B][T]")

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        availability_cpu = availability.detach().cpu()
        sequence_cpu = sequence_mask.detach().cpu()
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MaskManifest.FIELDS)
            writer.writeheader()
            for batch_index, sample_id in enumerate(sample_ids):
                if len(utterance_ids[batch_index]) != sequence_mask.shape[1]:
                    raise ValueError(
                        f"utterance_ids[{batch_index}] length does not match T"
                    )
                for time_index in range(sequence_mask.shape[1]):
                    if not bool(sequence_cpu[batch_index, time_index]):
                        continue
                    row = availability_cpu[batch_index, time_index].tolist()
                    if sum(row) == 0:
                        raise ValueError(
                            "manifest cannot contain an all-missing valid utterance"
                        )
                    writer.writerow(
                        {
                            "seed": int(seed),
                            "sample_id": sample_id,
                            "utterance_index": time_index,
                            "utterance_id": utterance_ids[batch_index][time_index],
                            "availability": json.dumps(
                                [bool(value) for value in row], separators=(",", ":")
                            ),
                            "mask_rate": float(mask_rate),
                            "sampler_version": sampler_version,
                        }
                    )

    @staticmethod
    def load(path: str) -> Dict[Tuple[str, int], Tuple[bool, bool, bool]]:
        result: Dict[Tuple[str, int], Tuple[bool, bool, bool]] = {}
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                values = json.loads(row["availability"])
                if len(values) != 3 or not any(values):
                    raise ValueError(f"invalid availability in manifest: {row}")
                key = (row["sample_id"], int(row["utterance_index"]))
                if key in result:
                    raise ValueError(f"duplicate manifest key: {key}")
                result[key] = tuple(bool(value) for value in values)
        return result

    @staticmethod
    def to_availability(
        manifest: Dict[Tuple[str, int], Tuple[bool, bool, bool]],
        sample_ids: Sequence[str],
        sequence_mask: torch.BoolTensor,
    ) -> torch.BoolTensor:
        _validate_inputs(sample_ids, sequence_mask)
        availability = torch.zeros(
            (*sequence_mask.shape, 3),
            dtype=torch.bool,
            device=sequence_mask.device,
        )
        for batch_index, sample_id in enumerate(sample_ids):
            for time_index in range(sequence_mask.shape[1]):
                if not bool(sequence_mask[batch_index, time_index]):
                    continue
                key = (sample_id, time_index)
                if key not in manifest:
                    raise KeyError(f"manifest missing key {key}")
                availability[batch_index, time_index] = torch.tensor(
                    manifest[key], dtype=torch.bool, device=availability.device
                )
        return availability

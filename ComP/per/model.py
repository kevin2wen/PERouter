from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..model import ComP
from .evidence import EvidenceEncoder
from .router import (
    PERRouter,
    evidence_target_confidence,
    fuse_candidates,
    gather_modality_summary,
    masked_route_probability,
)
from .types import (
    MODALITIES,
    EvidenceBatch,
    PEROutput,
)


class ComPPromptCandidateBuilder(nn.Module):
    """Own the ComP backbone and expose its stage-2 modality candidates."""

    def __init__(self, backbone: ComP):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        availability: torch.BoolTensor,
        fusion_fn=None,
        *,
        coordinator_mode: str = "on",
        coordinator_fixed_weights=None,
        fix_private_branch: bool = False,
    ):
        input_mask = availability.permute(1, 0, 2)
        umask = availability.any(dim=-1).to(torch.float32)
        return self.backbone(
            features,
            labels,
            input_mask,
            umask,
            False,
            return_intermediates=True,
            coordinator_mode=coordinator_mode,
            coordinator_fixed_weights=coordinator_fixed_weights,
            fix_private_branch=fix_private_branch,
            fusion_fn=fusion_fn,
        )


class PERStudent(nn.Module):
    """ComP plus external evidence and a missing-modality source router."""

    def __init__(
        self,
        args,
        adim: int,
        tdim: int,
        vdim: int,
        evidence_dim: int,
        top_k: int,
        *,
        task: str,
        use_router: bool = True,
        router_hidden: Optional[int] = None,
        router_dropout: float = 0.0,
        fix_private_branch: bool = True,
    ):
        super().__init__()
        if task not in {"regression", "classification"}:
            raise ValueError(f"unknown task: {task}")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.adim, self.tdim, self.vdim = int(adim), int(tdim), int(vdim)
        self.dim = int(args.hidden)
        self.evidence_dim = max(1, int(evidence_dim))
        self.top_k = int(top_k)
        self.task = task
        self.use_router = bool(use_router)
        self.per_mode = str(getattr(args, "per_mode", "task_router"))
        self.fixed_source_weight = float(getattr(args, "fixed_source_weight", 0.5))
        self.use_learned_router = self.use_router and self.per_mode != "fixed"
        if self.per_mode == "fixed" and not 0.0 <= self.fixed_source_weight <= 1.0:
            raise ValueError("fixed_source_weight must be in [0,1]")
        self.fix_private_branch = bool(fix_private_branch)
        self.coordinator_mode = getattr(args, "coordinator_mode", "on")
        self.coordinator_fixed_weights = getattr(
            args, "coordinator_fixed_weights", None
        )

        self.backbone = ComP(
            args,
            self.adim,
            self.tdim,
            self.vdim,
            self.dim,
            n_classes=args.n_classes,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=1,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            no_cuda=args.no_cuda,
            lbd=args.lbd,
        )
        self.candidate_builder = ComPPromptCandidateBuilder(self.backbone)
        if self.use_router:
            self.evidence_encoder = EvidenceEncoder(
                self.evidence_dim, self.dim, dropout=args.drop_rate
            )
            self.observed_projections = nn.ModuleDict(
                {
                    "audio": nn.Linear(self.adim, self.dim),
                    "text": nn.Linear(self.tdim, self.dim),
                    "video": nn.Linear(self.vdim, self.dim),
                }
            )
            self.router = (
                PERRouter(
                    self.dim,
                    hidden_dim=router_hidden,
                    dropout=router_dropout,
                )
                if self.use_learned_router
                else None
            )
        else:
            # B0/generated modes must not add unused router parameters.
            self.evidence_encoder = None
            self.observed_projections = None
            self.router = None

    def _empty_evidence_like(
        self,
        features: Dict[str, torch.Tensor],
        availability: torch.BoolTensor,
    ) -> EvidenceBatch:
        prototype = features[MODALITIES[0]]
        batch_size, seq_len = availability.shape[:2]
        return EvidenceBatch(
            ids=[
                [["" for _ in range(self.top_k)] for _ in range(seq_len)]
                for _ in range(batch_size)
            ],
            features=torch.zeros(
                batch_size,
                seq_len,
                self.top_k,
                self.evidence_dim,
                dtype=prototype.dtype,
                device=prototype.device,
            ),
            mask=torch.zeros(
                batch_size,
                seq_len,
                self.top_k,
                dtype=torch.bool,
                device=prototype.device,
            ),
            source_confidence=torch.zeros(
                batch_size,
                seq_len,
                self.top_k,
                dtype=prototype.dtype,
                device=prototype.device,
            ),
            conflict_score=torch.zeros(
                batch_size,
                seq_len,
                self.top_k,
                dtype=prototype.dtype,
                device=prototype.device,
            ),
        )

    def _pack_features(
        self,
        features: Dict[str, torch.Tensor],
        availability: torch.BoolTensor,
    ) -> torch.Tensor:
        batch_size, seq_len = availability.shape[:2]
        columns = []
        for modality_index, modality in enumerate(MODALITIES):
            value = features[modality]
            expected_dim = {
                "audio": self.adim,
                "text": self.tdim,
                "video": self.vdim,
            }[modality]
            if value.size(-1) != expected_dim:
                raise ValueError(
                    f"{modality} dim={value.size(-1)}, expected {expected_dim}"
                )
            if value.shape[:2] != (batch_size, seq_len):
                raise ValueError(
                    f"{modality} must share [B,T], got {tuple(value.shape)}"
                )
            if value.device != availability.device:
                raise ValueError(f"{modality} device does not match availability")
            keep = availability[:, :, modality_index].unsqueeze(-1)
            observed = torch.where(keep, value, torch.zeros_like(value))
            if not torch.isfinite(observed).all():
                raise FloatingPointError(
                    f"observed {modality} features contain non-finite values"
                )
            columns.append(observed.permute(1, 0, 2))
        return torch.cat(columns, dim=-1)

    def stage1_forward(
        self,
        features: Dict[str, torch.Tensor],
        availability: torch.BoolTensor,
        labels: torch.Tensor,
    ):
        packed = self._pack_features(features, availability)
        input_mask = availability.permute(1, 0, 2)
        umask = availability.any(dim=-1).to(torch.float32)
        return self.backbone(
            packed,
            labels,
            input_mask,
            umask,
            True,
            return_intermediates=True,
            coordinator_mode="off",
            fix_private_branch=self.fix_private_branch,
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        availability: torch.BoolTensor,
        evidence: Optional[EvidenceBatch] = None,
        return_intermediates: bool = True,
        route_override: Optional[torch.Tensor] = None,
        evidence_vectors_override: Optional[Dict[str, torch.Tensor]] = None,
    ) -> PEROutput:
        if availability.dtype != torch.bool:
            raise TypeError("availability must be torch.bool")
        if availability.ndim != 3 or availability.shape[-1] != 3:
            raise ValueError("availability must be [B,T,3]")
        batch_size, seq_len = availability.shape[:2]
        if evidence is None:
            evidence = self._empty_evidence_like(features, availability)
        evidence.validate(batch_size=batch_size, seq_len=seq_len)
        packed = self._pack_features(features, availability)
        if self.use_router:
            target_modalities = ~availability
            if evidence_vectors_override is None:
                evidence_vectors = self.evidence_encoder(
                    evidence, availability, target_modalities
                )
            else:
                if set(evidence_vectors_override) != set(MODALITIES):
                    raise KeyError('evidence_vectors_override must contain all modalities')
                evidence_vectors = {}
                for modality in MODALITIES:
                    value = evidence_vectors_override[modality]
                    expected = (batch_size, seq_len, self.dim)
                    if tuple(value.shape) != expected:
                        raise ValueError(
                            f'override {modality} shape={tuple(value.shape)}, expected {expected}'
                        )
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f'override {modality} is non-finite')
                    evidence_vectors[modality] = value.to(device=packed.device, dtype=packed.dtype)
            evidence_available = evidence.mask.any(dim=-1).unsqueeze(-1).expand(
                -1, -1, 3
            )
            confidence = evidence_target_confidence(
                evidence.mask, evidence.source_confidence
            )
            observed_summary = torch.stack(
                [
                    self.observed_projections[modality](
                        torch.where(
                            availability[:, :, index].unsqueeze(-1),
                            features[modality],
                            torch.zeros_like(features[modality]),
                        )
                    )
                    for index, modality in enumerate(MODALITIES)
                ],
                dim=2,
            )
        else:
            evidence_vectors = {
                modality: torch.zeros(
                    batch_size,
                    seq_len,
                    self.dim,
                    dtype=packed.dtype,
                    device=packed.device,
                )
                for modality in MODALITIES
            }
            evidence_available = torch.zeros(
                batch_size, seq_len, 3, dtype=torch.bool, device=packed.device
            )
            confidence = None
            observed_summary = None
        route_holder: Dict[str, torch.Tensor] = {
            "logits": torch.zeros(
                batch_size,
                seq_len,
                3,
                dtype=packed.dtype,
                device=packed.device,
            ),
            "route": torch.zeros(
                batch_size,
                seq_len,
                3,
                dtype=packed.dtype,
                device=packed.device,
            ),
        }

        def fuse(prompts, _input_mask):
            generated_summary = gather_modality_summary(prompts)
            if route_override is not None:
                if route_override.shape != availability.shape:
                    raise ValueError("route_override must match availability [B,T,3]")
                if not torch.isfinite(route_override).all():
                    raise FloatingPointError("route_override contains non-finite values")
                route_logits = torch.zeros_like(route_holder["route"])
                active = (~availability) & evidence_available
                route = route_override.to(route_holder["route"]) * active.to(
                    route_holder["route"].dtype
                )
            elif self.use_learned_router:
                route_logits = self.router(
                    observed_summary,
                    generated_summary,
                    gather_modality_summary(evidence_vectors),
                    availability,
                    confidence,
                )
                route = masked_route_probability(
                    route_logits, availability, evidence_available
                )
            else:
                route_logits = torch.zeros_like(route_holder["route"])
                active = (~availability) & evidence_available
                route = torch.full_like(
                    route_holder["route"], self.fixed_source_weight
                ) * active.to(route_holder["route"].dtype)
            route_holder["logits"] = route_logits
            route_holder["route"] = route
            return fuse_candidates(
                prompts,
                evidence_vectors,
                route,
                availability,
                evidence_available,
            )

        outputs = self.candidate_builder(
            packed,
            torch.zeros(batch_size, seq_len, dtype=packed.dtype, device=packed.device),
            availability,
            fusion_fn=fuse if self.use_router else None,
            coordinator_mode=self.coordinator_mode,
            coordinator_fixed_weights=self.coordinator_fixed_weights,
            fix_private_branch=self.fix_private_branch,
        )
        intermediates = outputs[-1]
        prompt_candidates = intermediates["prompt_candidates"]
        fused = intermediates["post_fusion_candidates"]
        if not self.use_router:
            fused = dict(prompt_candidates)
        for name in MODALITIES:
            if not torch.isfinite(prompt_candidates[name]).all():
                raise FloatingPointError(f"non-finite prompt candidate for {name}")
            if not torch.isfinite(fused[name]).all():
                raise FloatingPointError(f"non-finite fused candidate for {name}")
            if not torch.isfinite(evidence_vectors[name]).all():
                raise FloatingPointError(f"non-finite evidence candidate for {name}")
        if not torch.isfinite(outputs[1]).all():
            raise FloatingPointError("non-finite ComP logits")
        if not return_intermediates:
            intermediates = {}
        return PEROutput(
            logits=outputs[1],
            route_logits=route_holder["logits"],
            route=route_holder["route"],
            prompt_candidates=prompt_candidates,
            evidence=evidence_vectors,
            fused=fused,
            coordinator_weights=intermediates.get("coordinator_weights"),
            coordinator_logits=intermediates.get("coordinator_logits"),
            coordinator_mode=self.coordinator_mode,
            modality_logits=intermediates.get("modality_logits"),
            intermediates=intermediates,
        )

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn


def adapter_parameters(model: nn.Module) -> List[nn.Parameter]:
    encoder = getattr(model, "evidence_encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("model must expose an evidence_encoder module")
    return [parameter for parameter in encoder.parameters() if parameter.requires_grad]


def configure_adapter_training(model: nn.Module) -> nn.Module:
    """Freeze the complete model except the evidence encoder/projection."""
    encoder = getattr(model, "evidence_encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("model must expose an evidence_encoder module")
    model.requires_grad_(False)
    encoder.requires_grad_(True)
    model.eval()
    encoder.train()
    parameters = adapter_parameters(model)
    if not parameters:
        raise ValueError("evidence encoder has no trainable parameters")
    return encoder


def adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    encoder = getattr(model, "evidence_encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("model must expose an evidence_encoder module")
    return {key: value.detach().cpu() for key, value in encoder.state_dict().items()}


def adapter_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in adapter_parameters(model))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adapter_checkpoint(
    model: nn.Module,
    path: str,
    *,
    expected_source: Optional[str] = None,
) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("adapter checkpoint must be a dictionary")
    if expected_source is not None and payload.get("evidence_source") != expected_source:
        raise ValueError(
            f"adapter source={payload.get('evidence_source')} expected={expected_source}"
        )
    encoder = getattr(model, "evidence_encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("model must expose an evidence_encoder module")
    state = payload.get("adapter_state")
    if not isinstance(state, dict):
        raise ValueError("adapter checkpoint has no adapter_state")
    encoder.load_state_dict(state, strict=True)
    payload = dict(payload)
    payload["adapter_sha256"] = sha256_file(path)
    return payload

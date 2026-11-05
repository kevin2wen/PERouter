from __future__ import annotations

from types import SimpleNamespace

import torch

from ComP.per.model import PERStudent


def tiny_args(*, n_classes=1, coordinator_mode="on", device=None):
    device = device or torch.device("cpu")
    return SimpleNamespace(
        hidden=8,
        n_classes=n_classes,
        depth=2,
        num_heads=2,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        no_cuda=device.type == "cpu",
        lbd=0.3,
        seq_len=4,
        device=device,
        coordinator_mode=coordinator_mode,
        coordinator_fixed_weights=None,
    )


def tiny_model(
    *,
    n_classes=1,
    task="regression",
    use_router=True,
    coordinator_mode="on",
    evidence_dim=1,
    device=None,
):
    args = tiny_args(
        n_classes=n_classes, coordinator_mode=coordinator_mode, device=device
    )
    model = PERStudent(
        args,
        5,
        6,
        7,
        evidence_dim=evidence_dim,
        top_k=2,
        task=task,
        use_router=use_router,
        fix_private_branch=True,
    )
    return model.to(args.device)


def tiny_features(batch_size=2, seq_len=4, device=None):
    device = device or torch.device("cpu")
    return {
        "audio": torch.randn(batch_size, seq_len, 5, device=device),
        "text": torch.randn(batch_size, seq_len, 6, device=device),
        "video": torch.randn(batch_size, seq_len, 7, device=device),
    }


def text_availability(batch_size=2, seq_len=4, device=None):
    device = device or torch.device("cpu")
    availability = torch.ones(batch_size, seq_len, 3, dtype=torch.bool, device=device)
    availability[:, :, 1] = False
    return availability

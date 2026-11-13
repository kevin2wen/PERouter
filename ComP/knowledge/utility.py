from __future__ import annotations

import torch


def prediction_metrics(prediction, labels, skip_prediction, task: str):
    """Return (abs_error, correct, flip) tensors for oracle utilities."""
    if task == "regression":
        abs_error = (prediction - labels).abs()
        correct = (
            (prediction > 0) == (labels > 0)
        ).to(abs_error.dtype)
        flip = (
            (prediction > 0) != (skip_prediction > 0)
        ).to(abs_error.dtype)
        return abs_error, correct, flip
    abs_error = torch.zeros_like(labels, dtype=torch.float32)
    correct = (prediction == labels.long()).to(abs_error.dtype)
    flip = (prediction != skip_prediction).to(abs_error.dtype)
    return abs_error, correct, flip


def oracle_utility(
    prediction,
    labels,
    skip_prediction,
    *,
    task: str,
    objective: str,
    lambda_cls: float = 0.5,
    lambda_flip: float = 0.5,
):
    """Utility of a candidate; larger is better.

    ``task`` objective is handled by the caller because it needs the full
    output tensor. This helper covers the explicit diagnostic objectives.
    """
    abs_error, correct, flip = prediction_metrics(
        prediction, labels, skip_prediction, task
    )
    if objective == "mae":
        return -abs_error
    if objective == "acc":
        return correct - 1e-6 * abs_error
    if objective == "composite":
        return (
            -abs_error
            + float(lambda_cls) * correct
            - float(lambda_flip) * flip
        )
    raise ValueError(f"unsupported explicit objective: {objective}")


def regression_benefit(card_error, skip_error):
    """Benefit convention: positive means the card is better than skipping."""
    return skip_error - card_error

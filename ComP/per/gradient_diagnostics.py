from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch


def tensor_grad_norm(gradients: Iterable[Optional[torch.Tensor]]) -> float:
    values = [gradient.detach().float().pow(2).sum() for gradient in gradients if gradient is not None]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt())


def parameter_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return tensor_grad_norm(grads)


def component_gradient_norms(
    route_loss: torch.Tensor,
    task_loss: torch.Tensor,
    parameters: List[torch.nn.Parameter],
) -> Dict[str, float]:
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters:
        return {
            "route_component_grad_norm": 0.0,
            "task_component_grad_norm": 0.0,
            "route_to_task_grad_ratio": 0.0,
        }
    route_grads = torch.autograd.grad(
        route_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    task_grads = torch.autograd.grad(
        task_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    route_norm = tensor_grad_norm(route_grads)
    task_norm = tensor_grad_norm(task_grads)
    return {
        "route_component_grad_norm": route_norm,
        "task_component_grad_norm": task_norm,
        "route_to_task_grad_ratio": route_norm / max(task_norm, 1e-12),
    }

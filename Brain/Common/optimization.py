import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    base_lr: float = 1e-4,
) -> optim.AdamW:
    """
    Builds a specialized AdamW optimizer for models containing DAIN (dean) layers:
    1. Special layers in `dean` (mean_layer, scaling_layer, gating_layer) use dedicated
       learning rates (scaled by mean_lr, scale_lr, gate_lr) and 0.0 weight decay.
    2. LayerNorm/normalization layers, bias parameters, and `log_std` parameters are excluded from weight decay.
    3. Remaining model parameters receive standard weight decay.
    """
    dean_params_ids = set()
    if hasattr(model, "dean"):
        dean_params_ids.update(id(p) for p in model.dean.parameters())

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in dean_params_ids:
            continue
        if "norm" in name or name.endswith(".bias") or "log_std" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
    ]

    if hasattr(model, "dean"):
        param_groups.extend([
            {
                "params": list(model.dean.mean_layer.parameters()),
                "lr": base_lr * model.dean.mean_lr,
                "weight_decay": 0.0,
            },
            {
                "params": list(model.dean.scaling_layer.parameters()),
                "lr": base_lr * model.dean.scale_lr,
                "weight_decay": 0.0,
            },
            {
                "params": list(model.dean.gating_layer.parameters()),
                "lr": base_lr * model.dean.gate_lr,
                "weight_decay": 0.0,
            },
        ])

    return optim.AdamW(param_groups)


class opitmizar:
    """
    Backward-compatible wrapper for DAIN optimizer instantiation.
    """
    def __init__(
        self,
        net: nn.Module,
        learning_rate: float,
        lambda_l2: float,
        base_lr: float = 1e-4,
    ):
        self.net = net
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2
        self.base_lr = base_lr
        self.optimizer = build_optimizer(
            model=self.net,
            lr=self.learning_rate,
            weight_decay=self.lambda_l2,
            base_lr=self.base_lr,
        )

    def get_optimizer(self) -> optim.AdamW:
        return self.optimizer


# Proper alias for clean naming
Optimizer = opitmizar

        
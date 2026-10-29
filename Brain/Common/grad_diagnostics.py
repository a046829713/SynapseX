"""
Read-only gradient diagnostics for the PPO2 Mamba Actor-Critic.

All helpers in this module are strictly non-intrusive: they never mutate
``param.grad``, parameters, or optimizer state. They are used both by the
training loop telemetry (``PPO2_rl_train.update_ppo``) and by the offline
audit script (``tests/diagnose_ppo2_gradients.py``).
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

# Ordered (name-prefix, group) mapping. The first matching prefix wins.
MODULE_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("dean.mean_layer.", "dain_mean"),
    ("dean.scaling_layer.", "dain_scale"),
    ("dean.gating_layer.", "dain_gate"),
    ("time_embedding.", "input_fusion"),
    ("time_emb_projection.", "input_fusion"),
    ("market_embedding.", "input_fusion"),
    ("gate_layer.", "input_fusion"),
    ("feature_embedding.", "input_fusion"),
    ("mixer.", "backbone"),
    ("temporal_pool.", "pool"),
    ("portfolio_encoder.", "portfolio"),
    ("trunk_norm.", "trunk_norm"),
    ("fc_actor.", "actor"),
    ("fc_val.", "critic"),
    ("log_std", "log_std"),
)

DAIN_GROUPS = frozenset({"dain_mean", "dain_scale", "dain_gate"})

# Stable group ordering used for reporting.
GROUP_ORDER: Tuple[str, ...] = tuple(dict.fromkeys(g for _, g in MODULE_GROUPS))


def resolve_group(param_name: str) -> Optional[str]:
    """Returns the diagnostic group of a parameter name, or None if unmapped."""
    for prefix, group in MODULE_GROUPS:
        if param_name.startswith(prefix):
            return group
    return None


def build_param_groups(model: nn.Module) -> Dict[str, List[nn.Parameter]]:
    """
    Partitions all trainable parameters of ``model`` into diagnostic groups.

    Raises:
        AssertionError: if any trainable parameter is not covered by MODULE_GROUPS.
    """
    groups: Dict[str, List[nn.Parameter]] = {g: [] for g in GROUP_ORDER}
    unmapped: List[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = resolve_group(name)
        if group is None:
            unmapped.append(name)
            continue
        groups[group].append(param)

    assert not unmapped, (
        f"Gradient diagnostics: unmapped trainable parameters {unmapped}. "
        f"Extend MODULE_GROUPS in Brain/Common/grad_diagnostics.py."
    )
    n_grouped = sum(len(v) for v in groups.values())
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_grouped == n_trainable, (
        f"Gradient diagnostics: grouped {n_grouped} params but model has {n_trainable} trainable params."
    )
    return groups


def _sq_norm(params: Iterable[nn.Parameter], device: torch.device) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float32, device=device)
    for p in params:
        if p.grad is not None:
            total = total + torch.linalg.vector_norm(p.grad.detach().float()).pow(2)
    return total


def group_grad_norms(groups: Dict[str, List[nn.Parameter]]) -> Dict[str, float]:
    """
    Computes read-only per-group L2 gradient norms.

    Returns a dict with one entry per group plus aggregates:
      - ``dain``  : L2 norm over all DAIN groups
      - ``rest``  : L2 norm over all non-DAIN groups
      - ``total`` : L2 norm over all parameters (equals clip_grad_norm_'s return value)
    Parameters whose ``grad`` is None contribute 0.
    """
    device = None
    for params in groups.values():
        if params:
            device = params[0].device
            break
    if device is None:
        return {g: 0.0 for g in (*groups.keys(), "dain", "rest", "total")}

    with torch.no_grad():
        names = list(groups.keys())
        sq = torch.stack([_sq_norm(groups[g], device) for g in names])
        sq_cpu = sq.cpu().tolist()  # single host sync

    out: Dict[str, float] = {}
    dain_sq = 0.0
    rest_sq = 0.0
    for g, v in zip(names, sq_cpu):
        out[g] = float(v) ** 0.5
        if g in DAIN_GROUPS:
            dain_sq += v
        else:
            rest_sq += v
    out["dain"] = dain_sq ** 0.5
    out["rest"] = rest_sq ** 0.5
    out["total"] = (dain_sq + rest_sq) ** 0.5
    return out


def clip_coef(total_norm: float, max_norm: float) -> float:
    """Clip coefficient applied by torch.nn.utils.clip_grad_norm_ (clamped to <= 1)."""
    return float(min(1.0, max_norm / (total_norm + 1e-6)))


def adam_state_audit(
    groups: Dict[str, List[nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    eps: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Audits Adam/AdamW second-moment state per diagnostic group.

    For every parameter element computes the bias-corrected RMS gradient
    ``sqrt(exp_avg_sq / (1 - beta2^step))`` and reports per group:
      - p10 / p50 / p90 quantiles
      - frac_lt_eps   : fraction of elements with RMS < eps (update dominated by eps)
      - frac_lt_10eps : fraction of elements with RMS < 10 * eps
    Groups without optimizer state are reported with ``n_elements == 0``.
    """
    # Map parameter -> (beta2, eps) from its optimizer param_group.
    hp: Dict[int, Tuple[float, float]] = {}
    for pg in optimizer.param_groups:
        beta2 = float(pg["betas"][1])
        pg_eps = float(pg["eps"]) if eps is None else float(eps)
        for p in pg["params"]:
            hp[id(p)] = (beta2, pg_eps)

    report: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for g, params in groups.items():
            rms_chunks: List[torch.Tensor] = []
            ratio_chunks: List[torch.Tensor] = []
            for p in params:
                state = optimizer.state.get(p, None)
                if not state or "exp_avg_sq" not in state or id(p) not in hp:
                    continue
                beta2, p_eps = hp[id(p)]
                step = state.get("step", 0)
                step = float(step.item() if torch.is_tensor(step) else step)
                if step <= 0:
                    continue
                bc2 = 1.0 - beta2 ** step
                rms = (state["exp_avg_sq"].detach().float() / bc2).sqrt().flatten().cpu()
                rms_chunks.append(rms)
                ratio_chunks.append(rms / p_eps)

            if not rms_chunks:
                report[g] = {"n_elements": 0}
                continue
            rms_all = torch.cat(rms_chunks)
            ratio_all = torch.cat(ratio_chunks)
            # torch.quantile has an input size limit; subsample deterministically if needed.
            q_src = rms_all
            if q_src.numel() > 4_000_000:
                stride = q_src.numel() // 4_000_000 + 1
                q_src = q_src[::stride]
            q = torch.quantile(q_src, torch.tensor([0.1, 0.5, 0.9]))
            report[g] = {
                "n_elements": int(rms_all.numel()),
                "rms_p10": float(q[0]),
                "rms_p50": float(q[1]),
                "rms_p90": float(q[2]),
                "frac_lt_eps": float((ratio_all < 1.0).float().mean()),
                "frac_lt_10eps": float((ratio_all < 10.0).float().mean()),
            }
    return report

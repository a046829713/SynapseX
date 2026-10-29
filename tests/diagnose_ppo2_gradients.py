"""
Offline PPO2 gradient-norm & clip-strategy audit.

Spec: spec/feature/ppo2_gradient_norm_diagnostics.md

Analyses (all read-only w.r.t. the checkpoint on disk):
  A0  DAIN parameter drift from its identity initialisation.
  A1  Adam second-moment audit (eps-dominance) from the checkpoint optimizer state.
  A2  Per-module x per-loss-term gradient-norm attribution on fresh rollouts.
  A3  Volatility-tercile bias of DAIN gradient norms / clip coefficients.
  A4  Counterfactual one-step Adam update under 3 clip strategies
      (S0 global_clip, S1 exclude_dain, S2 no_clip).

Usage:
  ../bin/python tests/diagnose_ppo2_gradients.py \
      --checkpoint saves/<run>/checkpoint_iter_<N>.pt --rollouts 2 --seed 0 \
      [hydra overrides, e.g. training.NUM_WORKERS=4]
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hydra import compose, initialize_config_dir  # noqa: E402

from Brain.Common.grad_diagnostics import (  # noqa: E402
    DAIN_GROUPS,
    GROUP_ORDER,
    adam_state_audit,
    build_param_groups,
    clip_coef,
    group_grad_norms,
)
from PPO2_rl_train import PPO2  # noqa: E402

# Decision-matrix thresholds (spec section 5)
H1_DAIN_SHARE = 0.90
H2_VALUE_SHARE = 0.50
H3_EPS_FRAC = 0.25
H3_UPDATE_RATIO = 0.50
H4_VOL_RATIO = 3.0
SAFETY_KL_ABS = 0.05
SAFETY_KL_REL = 2.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def param_checksum(model: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([p.detach().flatten().double() for p in model.parameters()]).sum()


def collect_rollouts(ppo: PPO2, n_rollouts: int):
    """Collects n_rollouts * HORIZON_STEPS * NUM_WORKERS transitions, mirroring PPO2.train()."""
    cfg = ppo.cfg_training
    num_workers = ppo.num_workers
    gamma = float(cfg.GAMMA)
    total_steps = n_rollouts * int(cfg.HORIZON_STEPS)

    symbols = [ppo.symbol_queue.get_next() for _ in range(num_workers)]
    obs = ppo.vec_env.reset(symbols=symbols)
    ppo.buffer.clear()

    for _ in range(total_steps):
        out = ppo.agent.get_action(obs, stochastic=True)
        env_actions = out.clipped_action.cpu().numpy()
        next_obs, rewards, terminateds, truncateds, _infos = ppo.vec_env.step(env_actions)

        stored_rewards = rewards.copy()
        is_trunc = truncateds & (~terminateds)
        if np.any(is_trunc):
            idx = np.where(is_trunc)[0]
            trunc_obs = {
                "states": next_obs["states"][idx],
                "time_states": next_obs["time_states"][idx],
            }
            with torch.no_grad():
                tv = ppo.agent.get_action(trunc_obs, stochastic=False).value
            stored_rewards[idx] += gamma * tv.cpu().numpy().reshape(-1)

        dones = terminateds | truncateds
        ppo.buffer.store(
            obs, out.action.cpu(), out.log_prob.cpu(), stored_rewards,
            next_obs, dones, out.value.cpu(),
        )
        obs = next_obs
        for i, done in enumerate(dones):
            if done:
                new_obs_i = ppo.vec_env.reset_at(i, symbol=ppo.symbol_queue.get_next())
                obs["states"][i] = new_obs_i["states"]
                obs["time_states"][i] = new_obs_i["time_states"]

    with torch.no_grad():
        last_values = ppo.agent.get_action(obs, stochastic=False).value.cpu()
    ppo.buffer.compute_gae(last_values)

    # Pull the full flattened dataset (one batch, unshuffled)
    dataset_size = len(ppo.buffer.advantages)
    full = next(ppo.buffer.get_batches(batch_size=dataset_size, shuffle=False))
    keys = ("states", "time_states", "actions", "old_log_probs", "returns", "advantages", "old_values")
    data = {k: v.cpu() for k, v in zip(keys, full)}
    ppo.buffer.clear()
    return data


def take(data: Dict[str, torch.Tensor], idx: np.ndarray, device) -> Dict[str, torch.Tensor]:
    t_idx = torch.as_tensor(idx, dtype=torch.long)
    return {k: v[t_idx].to(device) for k, v in data.items()}


def compute_losses(ppo: PPO2, b: Dict[str, torch.Tensor], clip_eps: float):
    """Replicates the exact loss terms of PPO2.update_ppo for one minibatch."""
    log_probs, entropy, values = ppo.agent.evaluate_actions(
        b["states"], b["time_states"], b["actions"]
    )
    ratio = torch.exp(log_probs - b["old_log_probs"])
    adv = b["advantages"]
    mb_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    surr1 = ratio * mb_adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
    policy_loss = -torch.min(surr1, surr2).mean()

    v_clipped = b["old_values"] + torch.clamp(values - b["old_values"], -clip_eps, clip_eps)
    vf_unclipped = (values - b["returns"]).pow(2)
    vf_clipped = (v_clipped - b["returns"]).pow(2)
    value_loss = 0.5 * torch.max(vf_unclipped, vf_clipped).mean()
    entropy_loss = -entropy.mean()
    return policy_loss, value_loss, entropy_loss


def total_loss_backward(ppo: PPO2, b, cfg) -> None:
    pl, vl, el = compute_losses(ppo, b, float(cfg.CLIP_EPS))
    loss = pl + float(cfg.VF_COEF) * vl + float(cfg.ENT_COEF) * el
    ppo.optimizer.zero_grad(set_to_none=True)
    loss.backward()


def spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def window_volatility(states: torch.Tensor, close_idx: int) -> np.ndarray:
    """Per-sample std of 1-bar log-close differences over the observation window."""
    lc = states[:, :, close_idx].double()
    return torch.diff(lc, dim=1).std(dim=1).cpu().numpy()


def mean_dict(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    return {k: float(np.mean([r[k] for r in records])) for k in records[0]}


def fmt_row(name: str, values: List[float], width: int = 11) -> str:
    return f"{name:<14}" + "".join(f"{v:>{width}.4g}" for v in values)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def a0_dain_drift(model) -> Dict[str, float]:
    dean = model.dean
    with torch.no_grad():
        eye = torch.eye(dean.mean_layer.weight.shape[0], device=dean.mean_layer.weight.device)
        return {
            "mean_layer_minus_I_fro": float((dean.mean_layer.weight - eye).norm()),
            "scaling_layer_minus_I_fro": float((dean.scaling_layer.weight - eye).norm()),
            "gating_weight_fro": float(dean.gating_layer.weight.norm()),
            "gating_bias_l2": float(dean.gating_layer.bias.norm()),
        }


def a2_attribution(ppo, groups, data, cfg, n_mb: int, rng) -> Dict:
    device = ppo.device
    bs = int(cfg.BATCH_SIZE)
    n = data["advantages"].shape[0]
    perm = rng.permutation(n)
    n_mb = min(n_mb, n // bs)
    terms = ("policy", "value", "entropy", "total")
    rec: Dict[str, List[Dict[str, float]]] = {t: [] for t in terms}

    for m in range(n_mb):
        b = take(data, perm[m * bs:(m + 1) * bs], device)
        pl, vl, el = compute_losses(ppo, b, float(cfg.CLIP_EPS))
        weighted = {
            "policy": pl,
            "value": float(cfg.VF_COEF) * vl,
            "entropy": float(cfg.ENT_COEF) * el,
        }
        for t in ("policy", "value", "entropy"):
            ppo.optimizer.zero_grad(set_to_none=True)
            weighted[t].backward(retain_graph=True)
            rec[t].append(group_grad_norms(groups))
        ppo.optimizer.zero_grad(set_to_none=True)
        (weighted["policy"] + weighted["value"] + weighted["entropy"]).backward()
        rec["total"].append(group_grad_norms(groups))
    ppo.optimizer.zero_grad(set_to_none=True)

    out = {t: mean_dict(rec[t]) for t in terms}
    tot = out["total"]
    out["dain_share_sq"] = tot["dain"] ** 2 / max(tot["total"] ** 2, 1e-30)
    out["value_share_of_rest_sq"] = out["value"]["rest"] ** 2 / max(tot["rest"] ** 2, 1e-30)
    out["policy_share_of_rest_sq"] = out["policy"]["rest"] ** 2 / max(tot["rest"] ** 2, 1e-30)
    out["clip_coef_mean"] = float(np.mean(
        [clip_coef(r["total"], float(cfg.MAX_GRAD_NORM)) for r in rec["total"]]
    ))
    out["n_minibatches"] = n_mb
    return out


def a3_volatility(ppo, groups, data, cfg, repeats: int, close_idx: int, rng) -> Dict:
    device = ppo.device
    bs = int(cfg.BATCH_SIZE)
    sigma = window_volatility(data["states"], close_idx)
    q1, q2 = np.quantile(sigma, [1 / 3, 2 / 3])
    buckets = {
        "low": np.where(sigma <= q1)[0],
        "mid": np.where((sigma > q1) & (sigma <= q2))[0],
        "high": np.where(sigma > q2)[0],
    }
    out: Dict = {"sigma_q33": float(q1), "sigma_q67": float(q2)}
    mb_sigma, mb_dain = [], []
    for name, idx in buckets.items():
        recs = []
        if len(idx) < bs:
            out[name] = {"n_samples": int(len(idx)), "skipped": True}
            continue
        for _ in range(repeats):
            sel = rng.choice(idx, size=bs, replace=False)
            b = take(data, sel, device)
            total_loss_backward(ppo, b, cfg)
            gn = group_grad_norms(groups)
            gn["clip_coef"] = clip_coef(gn["total"], float(cfg.MAX_GRAD_NORM))
            recs.append(gn)
            mb_sigma.append(float(sigma[sel].mean()))
            mb_dain.append(gn["dain"])
        m = mean_dict(recs)
        out[name] = {
            "n_samples": int(len(idx)),
            "sigma_mean": float(sigma[idx].mean()),
            "dain": m["dain"],
            "rest": m["rest"],
            "total": m["total"],
            "clip_coef": m["clip_coef"],
        }
    ppo.optimizer.zero_grad(set_to_none=True)
    if "dain" in out.get("low", {}) and "dain" in out.get("high", {}):
        out["dain_low_over_high"] = out["low"]["dain"] / max(out["high"]["dain"], 1e-30)
    out["spearman_sigma_vs_dain"] = spearman(mb_sigma, mb_dain)
    return out


def a4_counterfactual(ppo, groups, data, cfg, n_mb: int, rng) -> Dict:
    device = ppo.device
    model = ppo.agent.model
    bs = int(cfg.BATCH_SIZE)
    max_norm = float(cfg.MAX_GRAD_NORM)
    all_params = [p for p in model.parameters() if p.requires_grad]
    rest_params = [p for g, ps in groups.items() if g not in DAIN_GROUPS for p in ps]
    group_of = {id(p): g for g, ps in groups.items() for p in ps}

    param_snapshot = [p.detach().clone() for p in model.parameters()]
    opt_snapshot = copy.deepcopy(ppo.optimizer.state_dict())
    checksum0 = param_checksum(model)

    def restore():
        with torch.no_grad():
            for p, s in zip(model.parameters(), param_snapshot):
                p.copy_(s)
        ppo.optimizer.load_state_dict(copy.deepcopy(opt_snapshot))
        ppo.optimizer.zero_grad(set_to_none=True)

    strategies = ("S0_global_clip", "S1_exclude_dain", "S2_no_clip")
    rec: Dict[str, List[Dict[str, float]]] = {s: [] for s in strategies}
    n = data["advantages"].shape[0]
    perm = rng.permutation(n)
    n_mb = min(n_mb, n // bs)

    for m in range(n_mb):
        b = take(data, perm[m * bs:(m + 1) * bs], device)
        with torch.no_grad():
            lp0, _, v0 = ppo.agent.evaluate_actions(b["states"], b["time_states"], b["actions"])
        for s in strategies:
            restore()
            total_loss_backward(ppo, b, cfg)
            if s == "S0_global_clip":
                torch.nn.utils.clip_grad_norm_(all_params, max_norm)
            elif s == "S1_exclude_dain":
                torch.nn.utils.clip_grad_norm_(rest_params, max_norm)
            ppo.optimizer.step()

            with torch.no_grad():
                sq = {g: 0.0 for g in GROUP_ORDER}
                for p, p0 in zip(model.parameters(), param_snapshot):
                    if id(p) in group_of:
                        sq[group_of[id(p)]] += float((p - p0).double().pow(2).sum())
                lp1, _, v1 = ppo.agent.evaluate_actions(
                    b["states"], b["time_states"], b["actions"]
                )
                log_r = lp1 - lp0
                approx_kl = float(((log_r.exp() - 1.0) - log_r).mean())
                dv = float((v1 - v0).abs().mean())
            r = {f"dtheta_{g}": v ** 0.5 for g, v in sq.items()}
            r["dtheta_dain"] = sum(v for g, v in sq.items() if g in DAIN_GROUPS) ** 0.5
            r["dtheta_rest"] = sum(v for g, v in sq.items() if g not in DAIN_GROUPS) ** 0.5
            r["approx_kl"] = approx_kl
            r["mean_abs_dvalue"] = dv
            rec[s].append(r)

    restore()
    checksum1 = param_checksum(model)
    assert torch.equal(checksum0, checksum1), "A4: parameters were not restored exactly"

    out = {s: mean_dict(rec[s]) for s in strategies}
    ref = max(out["S2_no_clip"]["dtheta_rest"], 1e-30)
    out["rest_update_ratio_S0_over_S2"] = out["S0_global_clip"]["dtheta_rest"] / ref
    out["rest_update_ratio_S1_over_S2"] = out["S1_exclude_dain"]["dtheta_rest"] / ref
    out["n_minibatches"] = n_mb
    out["params_restored_exactly"] = True
    return out


def verdicts(report: Dict) -> Dict:
    a1, a2, a3, a4 = report.get("A1"), report["A2"], report["A3"], report.get("A4")
    v: Dict = {}
    v["H1_dain_dominates"] = {
        "metric": a2["dain_share_sq"], "threshold": f">= {H1_DAIN_SHARE}",
        "triggered": a2["dain_share_sq"] >= H1_DAIN_SHARE,
    }
    v["H2_critic_dominates_rest"] = {
        "metric": a2["value_share_of_rest_sq"], "threshold": f">= {H2_VALUE_SHARE}",
        "triggered": a2["value_share_of_rest_sq"] >= H2_VALUE_SHARE,
    }
    eps_frac = a1.get("rest_frac_lt_eps") if a1 else None
    ratio = a4.get("rest_update_ratio_S0_over_S2") if a4 else None
    h3 = (eps_frac is not None and eps_frac >= H3_EPS_FRAC) or (
        ratio is not None and ratio <= H3_UPDATE_RATIO
    )
    v["H3_clip_suppresses_learning"] = {
        "metric_eps_frac": eps_frac, "metric_update_ratio_S0_over_S2": ratio,
        "threshold": f"eps_frac >= {H3_EPS_FRAC} or ratio <= {H3_UPDATE_RATIO}",
        "triggered": bool(h3),
    }
    vol_ratio = a3.get("dain_low_over_high")
    v["H4_volatility_reweighting"] = {
        "metric": vol_ratio, "threshold": f">= {H4_VOL_RATIO}",
        "triggered": vol_ratio is not None and vol_ratio >= H4_VOL_RATIO,
    }
    if a4:
        kl2 = a4["S2_no_clip"]["approx_kl"]
        kl1 = a4["S1_exclude_dain"]["approx_kl"]
        unsafe = kl2 > SAFETY_KL_ABS or kl2 > SAFETY_KL_REL * max(kl1, 1e-12)
        v["Safety_no_clip_unsafe"] = {
            "kl_S2": kl2, "kl_S1": kl1,
            "threshold": f"kl_S2 > {SAFETY_KL_ABS} or kl_S2 > {SAFETY_KL_REL} x kl_S1",
            "triggered": bool(unsafe),
        }
    return v


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint_iter_*.pt (model + optimizer state).")
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attr-minibatches", type=int, default=16)
    parser.add_argument("--vol-repeats", type=int, default=8)
    parser.add_argument("--cf-minibatches", type=int, default=8)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. training.NUM_WORKERS=4")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=args.overrides)

    t0 = time.time()
    ppo = PPO2(cfg)
    try:
        model = ppo.agent.model
        report: Dict = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "overrides": args.overrides,
            "max_grad_norm": float(ppo.cfg_training.MAX_GRAD_NORM),
            "batch_size": int(ppo.cfg_training.BATCH_SIZE),
        }

        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=ppo.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            ppo.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            report["checkpoint_iteration"] = ckpt.get("iteration")
            report["checkpoint_total_steps"] = ckpt.get("total_steps")
        else:
            print("[WARN] No --checkpoint given: using randomly initialised model with empty Adam state. "
                  "A1 and A4 are skipped (smoke-test mode only).")

        groups = build_param_groups(model)
        report["group_param_counts"] = {g: int(sum(p.numel() for p in ps)) for g, ps in groups.items()}

        # A0
        report["A0"] = a0_dain_drift(model)

        # A1
        if args.checkpoint:
            audit = adam_state_audit(groups, ppo.optimizer)
            n_rest = sum(a.get("n_elements", 0) for g, a in audit.items() if g not in DAIN_GROUPS)
            lt = sum(a.get("n_elements", 0) * a.get("frac_lt_eps", 0.0)
                     for g, a in audit.items() if g not in DAIN_GROUPS)
            lt10 = sum(a.get("n_elements", 0) * a.get("frac_lt_10eps", 0.0)
                       for g, a in audit.items() if g not in DAIN_GROUPS)
            report["A1"] = {
                "per_group": audit,
                "rest_frac_lt_eps": lt / max(n_rest, 1),
                "rest_frac_lt_10eps": lt10 / max(n_rest, 1),
            }

        # Rollouts
        print(f"[Audit] Collecting {args.rollouts} rollout(s)...")
        data = collect_rollouts(ppo, args.rollouts)
        report["n_transitions"] = int(data["advantages"].shape[0])

        close_idx = 3  # 'log_close' position in State_time_step_template.info_list
        assert ppo.vec_env.observation_space["states"].shape[1] > close_idx

        print("[Audit] A2 attribution...")
        report["A2"] = a2_attribution(ppo, groups, data, ppo.cfg_training, args.attr_minibatches, rng)
        print("[Audit] A3 volatility buckets...")
        report["A3"] = a3_volatility(ppo, groups, data, ppo.cfg_training, args.vol_repeats, close_idx, rng)
        if args.checkpoint:
            print("[Audit] A4 counterfactual clip strategies...")
            report["A4"] = a4_counterfactual(ppo, groups, data, ppo.cfg_training, args.cf_minibatches, rng)

        report["verdicts"] = verdicts(report)
        report["elapsed_sec"] = time.time() - t0
        print_summary(report)

        out_path = args.out or os.path.join(
            REPO_ROOT, "results", "grad_diagnostics",
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"\n[Audit] Report written to: {out_path}")
    finally:
        ppo.vec_env.close()


def print_summary(r: Dict):
    bar = "=" * 100
    print(f"\n{bar}\nPPO2 GRADIENT AUDIT  (max_grad_norm={r['max_grad_norm']}, batch={r['batch_size']}, "
          f"transitions={r['n_transitions']})\n{bar}")

    print("\n[A0] DAIN drift from init:")
    for k, v in r["A0"].items():
        print(f"  {k:<28} {v:.6g}")

    if "A1" in r:
        print("\n[A1] Adam sqrt(v_hat) per group (eps-dominance):")
        print(f"  {'group':<14}{'n':>11}{'p10':>11}{'p50':>11}{'p90':>11}{'<eps':>11}{'<10eps':>11}")
        for g, a in r["A1"]["per_group"].items():
            if a.get("n_elements", 0) == 0:
                print(f"  {g:<14}{'n/a':>11}")
                continue
            print("  " + fmt_row(g, [a["n_elements"], a["rms_p10"], a["rms_p50"], a["rms_p90"],
                                     a["frac_lt_eps"], a["frac_lt_10eps"]]))
        print(f"  non-DAIN frac < eps: {r['A1']['rest_frac_lt_eps']:.4f} | "
              f"< 10eps: {r['A1']['rest_frac_lt_10eps']:.4f}")

    a2 = r["A2"]
    cols = list(GROUP_ORDER) + ["dain", "rest", "total"]
    print(f"\n[A2] Pre-clip grad norm by module x loss term (mean of {a2['n_minibatches']} minibatches):")
    print(f"  {'term':<14}" + "".join(f"{c[:10]:>11}" for c in cols))
    for t in ("policy", "value", "entropy", "total"):
        print("  " + fmt_row(t, [a2[t][c] for c in cols]))
    print(f"  DAIN share of total^2: {a2['dain_share_sq']:.4f} | value share of rest^2: "
          f"{a2['value_share_of_rest_sq']:.4f} | policy share of rest^2: {a2['policy_share_of_rest_sq']:.4f} | "
          f"clip coef: {a2['clip_coef_mean']:.5f}")

    a3 = r["A3"]
    print("\n[A3] Volatility terciles (sigma = std of 1-bar log-close diff):")
    print(f"  {'bucket':<14}{'n':>11}{'sigma':>11}{'dain':>11}{'rest':>11}{'total':>11}{'clipcoef':>11}")
    for name in ("low", "mid", "high"):
        b = a3.get(name, {})
        if b.get("skipped") or "dain" not in b:
            print(f"  {name:<14}{'skipped':>11}")
            continue
        print("  " + fmt_row(name, [b["n_samples"], b["sigma_mean"], b["dain"], b["rest"],
                                    b["total"], b["clip_coef"]]))
    print(f"  DAIN low/high: {a3.get('dain_low_over_high', float('nan')):.3f} | "
          f"Spearman(sigma, dain): {a3['spearman_sigma_vs_dain']:+.3f}")

    if "A4" in r:
        a4 = r["A4"]
        print(f"\n[A4] Counterfactual one-step update (mean of {a4['n_minibatches']} minibatches):")
        print(f"  {'strategy':<18}{'|dθ_rest|':>12}{'|dθ_dain|':>12}{'|dθ_actor|':>12}"
              f"{'|dθ_critic|':>12}{'approxKL':>12}{'|dV|':>12}")
        for s in ("S0_global_clip", "S1_exclude_dain", "S2_no_clip"):
            x = a4[s]
            print(f"  {s:<18}" + "".join(f"{v:>12.4g}" for v in (
                x["dtheta_rest"], x["dtheta_dain"], x["dtheta_actor"], x["dtheta_critic"],
                x["approx_kl"], x["mean_abs_dvalue"])))
        print(f"  rest update ratio S0/S2: {a4['rest_update_ratio_S0_over_S2']:.4f} | "
              f"S1/S2: {a4['rest_update_ratio_S1_over_S2']:.4f}")

    print("\n[Verdicts]")
    for k, v in r["verdicts"].items():
        flag = "TRIGGERED" if v["triggered"] else "not triggered"
        metrics = {mk: mv for mk, mv in v.items() if mk not in ("threshold", "triggered")}
        mstr = ", ".join(f"{mk}={mv:.4g}" if isinstance(mv, float) else f"{mk}={mv}"
                         for mk, mv in metrics.items())
        print(f"  {k:<30} {flag:<14} ({mstr}; {v['threshold']})")


if __name__ == "__main__":
    main()

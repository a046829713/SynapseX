# Specification: PPO2 Per-Module Gradient Norm Diagnostics & Clip-Strategy Audit

- **Category**: `feature`
- **Spec Path**: `spec/feature/ppo2_gradient_norm_diagnostics.md`
- **Status**: `Pending Review`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)
5. [5. Decision Matrix (Diagnosis → Follow-up Spec)](#5-decision-matrix-diagnosis--follow-up-spec)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
At ~34M steps (Iter ~8390) the PPO2 run reports `GradNorm ≈ 200–290` while
`MAX_GRAD_NORM = 0.5`. Every minibatch is therefore globally rescaled by roughly `1/400 – 1/580`.
The user hypothesises that the DAIN layer produces the dominant gradients and that this global
clipping is one reason the policy fails to converge (`NetPnL(100) ≈ 0`, `Win(100) ≈ 50–54%`,
`ExplainedVariance ≈ 0.13–0.33`, `KL ≈ 0.003–0.012`).

This hypothesis is **plausible but unverified**. The current telemetry only logs the single
pre-clip global norm, so we cannot tell which module owns that norm, whether the clip actually
suppresses the effective Adam update, or what would happen if the clip were removed or restricted.

**Goal of this spec**: add *measurement only*. No training behaviour changes. The output is a set
of numbers that select the follow-up fix (see §5).

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **Telemetry** | Only the global pre-clip `GradNorm` (minibatch mean) is logged. | Per-module pre-clip norms (`dean`, backbone, actor, critic, …), DAIN vs. rest split, and the mean clip coefficient are logged every iteration. |
| **Attribution** | Unknown whether the norm comes from DAIN, critic (reward ×100), or the actor. | Norm is decomposed by module **and** by loss term (policy / value / entropy) in an offline audit. |
| **Clip impact** | Unknown whether global clipping shrinks non-DAIN updates under AdamW `eps=1e-5`. | Offline audit measures the eps-dominated parameter fraction and the counterfactual one-step Adam update size / KL under 3 clip strategies. |
| **Training dynamics** | — | **Unchanged.** `clip_grad_norm_` call, optimizer, learning rates, and loss are untouched. |

### 1.3 Technical Root Cause & References
- Global clip: [`PPO2_rl_train.py:L320-L325`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L320-L325)
  — `clip_grad_norm_` computes one L2 norm over **all** parameters and multiplies **all** gradients by
  `min(1, max_norm / total_norm)`. A large DAIN gradient therefore shrinks Mamba / actor / critic
  gradients that are themselves small.
- DAIN learning rates: [`optimization.py:L7-L60`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py#L7-L60)
  combined with [`dain.py:L8`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py#L8)
  give `mean_layer lr = 1e-4 × 1e-5 = 1e-9`, `scaling_layer lr = 1e-9`, `gating_layer lr = 1e-7`.
  DAIN is effectively frozen, so its gradients carry no useful learning signal but still enter the
  global norm.
- Likely gradient source: [`dain.py:L74-L93`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py#L74-L93)
  — `x / adaptive_std` on raw `log_open/high/low/close/volume` windows. For low-volatility windows
  `std` is small, so `∂L/∂W_scale ∝ 1/std²` explodes. Hypothesis: clip coefficient is
  volatility-dependent, which silently down-weights low-volatility samples in the policy/value update.
- AdamW scale-invariance caveat: Adam largely cancels a constant gradient rescale, **except** via
  `eps=1e-5` ([`optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py#L12)).
  If clipped per-parameter gradient RMS approaches `eps`, the effective step shrinks. This must be
  measured, not assumed.
- Telemetry print: [`PPO2_rl_train.py:L548-L558`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L548-L558).
  Note: the user's runtime log uses labels `ValueLoss / Entropy / ExplainedVariance`, while the repo
  prints `VfLoss / Ent / ExplVar`. The running container (`/SynapseX`) may be on a different
  revision; this must be confirmed before comparing numbers.

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **1** | `Brain/Common/grad_diagnostics.py` (new) | **In-Scope** | Pure helpers: module grouping, per-group grad norms, clip-coefficient, Adam-state audit. No side effects on gradients. |
| **2** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **In-Scope** | In `update_ppo`, compute per-group norms after `backward()` and **before** the unchanged `clip_grad_norm_`; aggregate and print one extra telemetry segment. |
| **3** | `tests/diagnose_ppo2_gradients.py` (new) | **In-Scope** | Offline audit script: loads a checkpoint, collects rollouts, runs attribution / volatility / counterfactual analyses, writes JSON report. |
| **4** | `tests/test_grad_diagnostics.py` (new) | **In-Scope** | Unit tests for helper correctness and non-intrusiveness. |
| **5** | [`Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py), [`model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py), [`dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py) | **Out-of-Scope** | No architecture / forward-pass change. |
| **6** | [`optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py), [`ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **Out-of-Scope** | No LR, eps, `MAX_GRAD_NORM` or YAML key change. |
| **7** | [`environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py), [`experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) | **Out-of-Scope** | Reused read-only by the offline script. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Telemetry must be read-only w.r.t. `param.grad` (no in-place ops, no `.grad` reassignment).
  - The existing `clip_grad_norm_(self.agent.model.parameters(), max_grad_norm)` call stays byte-identical.
  - Checkpoint format (`model_state_dict`, `optimizer_state_dict`) unchanged; old checkpoints remain loadable.
- **Out-of-Scope (Non-Goals)**:
  - Removing, restricting, or retuning gradient clipping (decided in a **follow-up spec** per §5).
  - Freezing DAIN, changing DAIN inputs, or changing reward scale / `VF_COEF`.
  - TensorBoard integration (console log + JSON only).

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [ ] **Step 1: `Brain/Common/grad_diagnostics.py` helpers**
  - [ ] `MODULE_GROUPS`: ordered prefix → group map (see 3.2.1). Every trainable parameter must map to exactly one group; unmapped names raise `AssertionError`.
  - [ ] `group_grad_norms(model) -> Dict[str, float]`: per-group L2 norm of `.grad` (params with `grad is None` contribute 0), plus `"dain"`, `"rest"`, `"total"` aggregates. Wrapped in `torch.no_grad()`, computed with `torch.linalg.vector_norm` on detached grads; single host sync per call.
  - [ ] `clip_coef(total_norm, max_norm) -> float` = `min(1.0, max_norm / (total_norm + 1e-6))` (matches PyTorch semantics).
  - [ ] `adam_state_audit(model, optimizer, eps) -> Dict`: per group, quantiles (p10/p50/p90) of `sqrt(exp_avg_sq / bias_correction2)` and fractions `< eps`, `< 10·eps`.
- [ ] **Step 2: Training telemetry in `update_ppo`**
  - [ ] After `total_loss.backward()` and before `clip_grad_norm_`, call `group_grad_norms`; append to per-iteration lists.
  - [ ] Return new metrics: `grad_norm_dain`, `grad_norm_rest`, `grad_norm_actor`, `grad_norm_critic`, `grad_norm_backbone`, `clip_coef_mean`, `clip_coef_min`.
  - [ ] Extend the iteration print with: `GN[dain/rest/actor/critic]: 281.2/0.91/0.12/0.85 | ClipCoef: 0.0018`.
- [ ] **Step 3: Offline audit script `tests/diagnose_ppo2_gradients.py`**
  - [ ] CLI: `--checkpoint`, `--rollouts` (default 2), `--seed` (default 0), `--out` (default `results/grad_diagnostics/<timestamp>.json`).
  - [ ] Build `PPO2` via Hydra compose (same `configs/config.yaml`), load `model_state_dict` + `optimizer_state_dict` (strict).
  - [ ] **A0 – DAIN drift**: `‖W_mean − I‖_F`, `‖W_scale − I‖_F`, `‖W_gate‖_F`, `‖b_gate‖` (confirms frozen DAIN).
  - [ ] **A1 – Adam-state audit** from the checkpoint optimizer (`adam_state_audit`). Because `exp_avg_sq` was accumulated from *post-clip* gradients, this directly measures eps-dominance under the current regime.
  - [ ] **A2 – Attribution**: collect `--rollouts × NUM_WORKERS × HORIZON_STEPS` transitions with `stochastic=True`, compute GAE via `RolloutBuffer`. For every minibatch (`BATCH_SIZE`), replicate the exact loss of `update_ppo` and run three separate backward passes (policy, `vf_coef·value`, `ent_coef·entropy`) plus the combined one; record per-group norms per loss term. No `optimizer.step()`.
  - [ ] **A3 – Volatility bias**: per sample compute window volatility `σ = std(diff(log_close))` over the 300-bar window (feature index of `log_close` resolved from `info_list` in [`env_components.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L24)). Split into terciles; build `BATCH_SIZE` minibatches per tercile (≥8 repeats); report mean DAIN norm, rest norm and clip coefficient per tercile, plus Spearman ρ(σ, DAIN norm) at minibatch level.
  - [ ] **A4 – Counterfactual one-step update**: for ≥8 minibatches, from the identical checkpoint parameters + deep-copied optimizer state, apply one `optimizer.step()` under each strategy, then restore:
    - `S0 global_clip` (current): clip all params to `MAX_GRAD_NORM`.
    - `S1 exclude_dain`: clip non-DAIN params to `MAX_GRAD_NORM`, leave DAIN unclipped.
    - `S2 no_clip`: no clipping.
    Record per-group `‖Δθ‖`, ratio `‖Δθ_rest‖(S) / ‖Δθ_rest‖(S2)`, and post-step `approx_kl` + `|Δvalue|` on the same minibatch.
  - [ ] Print summary tables; dump everything to JSON; print §5 verdict flags (H1–H4, Safety).
- [ ] **Step 4: Clean up & Defensiveness**
  - [ ] Assert group coverage equals the full trainable parameter set.
  - [ ] Assert offline script restores parameters exactly between strategies (`torch.equal` on a parameter checksum).
  - [ ] English-only comments/docstrings; no dead code.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 `Brain/Common/grad_diagnostics.py`
```python
MODULE_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("dean.mean_layer.",    "dain_mean"),
    ("dean.scaling_layer.", "dain_scale"),
    ("dean.gating_layer.",  "dain_gate"),
    ("time_embedding.",     "input_fusion"),
    ("time_emb_projection.","input_fusion"),
    ("market_embedding.",   "input_fusion"),
    ("gate_layer.",         "input_fusion"),
    ("feature_embedding.",  "input_fusion"),
    ("mixer.",              "backbone"),
    ("temporal_pool.",      "pool"),
    ("portfolio_encoder.",  "portfolio"),
    ("trunk_norm.",         "trunk_norm"),
    ("fc_actor.",           "actor"),
    ("fc_val.",             "critic"),
    ("log_std",             "log_std"),
)
DAIN_GROUPS = {"dain_mean", "dain_scale", "dain_gate"}

def build_param_groups(model: nn.Module) -> Dict[str, List[nn.Parameter]]: ...
def group_grad_norms(groups: Dict[str, List[nn.Parameter]]) -> Dict[str, float]:
    """Read-only per-group L2 grad norms + 'dain', 'rest', 'total' aggregates."""
def clip_coef(total_norm: float, max_norm: float) -> float: ...
def adam_state_audit(groups, optimizer: torch.optim.Optimizer, eps: float) -> Dict[str, Dict[str, float]]: ...
```

#### 3.2.2 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L320-L325)
```python
self.optimizer.zero_grad()
total_loss.backward()
gn = group_grad_norms(self._param_groups)          # NEW, read-only
grad_norm = torch.nn.utils.clip_grad_norm_(        # UNCHANGED
    self.agent.model.parameters(), max_grad_norm
)
self.optimizer.step()
```
`self._param_groups = build_param_groups(self.agent.model)` is created once in `_prepare_agent`.

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST use the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python` · Relative path: `../bin/python`

```bash
# Unit tests
../bin/python -m pytest tests/test_grad_diagnostics.py -v

# Offline audit on the latest checkpoint
../bin/python tests/diagnose_ppo2_gradients.py \
    --checkpoint saves/<run>/checkpoint_iter_<N>.pt --rollouts 2 --seed 0

# Telemetry smoke run (Ctrl+C after ~5 iterations)
../bin/python PPO2_rl_train.py
```

### 4.2 Acceptance Verification Checklist
- [ ] **1. Functional Correctness**
  - [ ] `sqrt(Σ group_norm²)` equals the value returned by `clip_grad_norm_` within relative tolerance `1e-4`.
  - [ ] Group coverage: every trainable parameter assigned exactly once (unit test on the real `MambaContinuousActorCritic`).
  - [ ] `clip_coef` matches PyTorch's internal coefficient on a toy model (compare grads after `clip_grad_norm_` with `grad × coef`, tol `1e-6`).
- [ ] **2. Non-Intrusiveness**
  - [ ] `param.grad` tensors are bit-identical (`torch.equal`) before and after `group_grad_norms`.
  - [ ] With fixed seed, one `update_ppo` call on a synthetic buffer yields bit-identical parameters with and without telemetry.
  - [ ] Offline script: parameter checksum identical before and after each A4 strategy.
- [ ] **3. Boundary & Edge Cases**
  - [ ] Params with `grad is None` contribute 0 and do not raise.
  - [ ] Adam audit handles params without optimizer state (step 0) gracefully (reported as `n/a`).
- [ ] **4. Runtime & Empirical**
  - [ ] Training FPS drop ≤ 2% versus baseline over ≥5 iterations.
  - [ ] Offline script completes on a real checkpoint, prints H1–H4 / Safety flags, writes JSON.

---

## 5. Decision Matrix (Diagnosis → Follow-up Spec)

Thresholds are evaluated on the A1–A4 outputs of the offline audit (mean over minibatches).

| ID | Hypothesis | Metric | Trigger | Follow-up |
| :---: | :--- | :--- | :--- | :--- |
| **H1** | DAIN dominates global norm | `dain² / total²` | ≥ 0.90 | Spec: clip only non-DAIN params (S1) **or** freeze DAIN (`requires_grad=False`) |
| **H2** | Critic dominates non-DAIN norm | value-term share of `rest²` | ≥ 0.50 | Spec: value target scaling / `VF_COEF` / value normalisation |
| **H3** | Global clip suppresses learning via Adam `eps` | non-DAIN fraction `sqrt(v̂) < eps` **or** `‖Δθ_rest‖(S0)/‖Δθ_rest‖(S2)` | ≥ 0.25 **or** ≤ 0.5 | Confirms clipping change is worth doing |
| **H4** | Volatility-dependent sample re-weighting | DAIN norm (low-σ tercile) / (high-σ tercile) | ≥ 3.0 | Also motivates fixing DAIN inputs (separate spec) |
| **Safety** | Removing clip entirely is unsafe | single-step `approx_kl` under S2 | > 0.05 **or** > 2× S1 | Do **not** adopt S2; prefer S1 |

If none of H1–H3 trigger, gradient clipping is **not** a material cause of non-convergence and
effort should move to policy noise / value-function quality (see prior analysis).

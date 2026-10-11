# Specification: Multi-Instrument PPO2 Training with Custom Lightweight SubprocVecEnv

- **Category**: `feature`
- **Spec Path**: `spec/feature/ppo2_multiprocess_vector_env.md`
- **Status**: `Implemented & Verified`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
The current [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) training pipeline operates on a **single environment, single instrument sequential mode**. During each iteration, the agent interacts sequentially with 1 environment for `N_STEPS = 1000` steps. Although [`TrainingEnv.reset()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L353) randomly samples an instrument upon completion, this structure suffers from multiple severe bottlenecks:
1. **Low Sample Throughput & GPU Underutilization**: The host machine possesses an NVIDIA GeForce RTX 2060 GPU, 12 CPU cores, and ~16 GB RAM. Benchmarks demonstrate that batched forward passes on [`MambaContinuousActorCritic`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L257) with batch size $N=8$ require only ~5.13 ms per step and ~124.6 MB VRAM. Running single-step interaction sequentially on CPU starves the GPU and limits training throughput.
2. **Poor Instrument Coverage & Representation Bias**: The dataset in [`Brain/simulation/train_data`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/simulation/train_data) contains **179 unique instrument CSV files** (cryptos, equities, indices, commodities). Random single-worker selection takes thousands of iterations to achieve adequate sample coverage and risks severe representation imbalance across symbols.
3. **Memory Risk with Naive Subprocessing**: Each parsed dataset takes ~15–30 MB in memory (~2.8 GB total for all 179 instruments). Spawning multiple workers without a unified shared memory preloader would lead each worker process to load symbols independently, consuming up to $8 \times 2.8\text{ GB} > 22\text{ GB}$, triggering an out-of-memory (OOM) crash on a 16 GB machine.

### 1.2 User Alignment & Design Decisions
Following user consultation and confirmation:
1. **Number of Parallel Workers**: $N = 8$ parallel environments. This matches the host system's 12 CPU cores closely, maximizing throughput without CPU core oversubscription or context switching penalties.
2. **Instrument Rotation Strategy**: **Option A: Global Shuffled Round-Robin Queue**. All 179 unique instruments are shuffled into a global queue. Whenever any worker triggers an episode termination (`done=True`), the coordinator pops the next instrument from the queue and resets that worker to the new instrument, guaranteeing 100% uniform coverage across all 179 symbols.
3. **Environment Architecture**: **Option A: Custom Lightweight SubprocVecEnv** using native Python `multiprocessing.Pipe`. Avoids Gymnasium version discrepancies and auto-reset observation ambiguities with Dict observation spaces while giving fine-grained programmatic control over instrument resets.
4. **Rollout Horizon & Sample Budget**: **Total 4,000 steps per PPO iteration**. With $N = 8$ workers, each worker collects $T = 500$ steps per iteration ($500 \times 8 = 4,000$ steps).

### 1.3 Current vs. Expected Behavior
| Dimension | Current Behavior ([`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)) | Expected Behavior (Vectorized Multi-Instrument) |
| :--- | :--- | :--- |
| **Worker Concurrency** | 1 sequential worker ($N=1$). | 8 parallel worker processes ($N=8$) managed by lightweight `SubprocVecEnv`. |
| **Rollout Step Count** | 1,000 steps per update on a single trajectory. | $T=500$ steps per worker $\times 8$ workers = 4,000 transitions per PPO iteration. |
| **Symbol Coverage** | Single symbol picked via `np.random.choice` per episode. | Global shuffled queue cycling through all 179 symbols sequentially on each worker reset. |
| **Data Memory** | Child/single env dynamically loads CSVs on demand. | Main process preloads all 179 symbols into memory once; workers share read-only data via OS CoW / reference. |
| **Agent Action Inference** | Single input `squeeze(0)` in [`PPO2Agent.get_action`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L80). | Batched tensor input `(N, seq_dim, feat_dim)` producing actions `(N, 1)`, values `(N,)`, log_probs `(N,)`. |
| **Rollout Buffer & GAE** | 1D list of transitions; 1D GAE loop in [`RolloutBuffer`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py#L35). | 2D array/tensor buffer of shape `[T, N, ...]`; 2D vectorized GAE masking transitions per worker, flattened to `[T*N, ...]`. |

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | Add `NUM_WORKERS: 8`, `HORIZON_STEPS: 500`, update `N_STEPS: 4000` and rollout configurations. |
| **Item 2** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **In-Scope** | 1. Add `preloaded_data` argument to [`TrainingEnv`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L315) to enable memory sharing.<br>2. Implement lightweight `SubprocVecEnv` using `multiprocessing.Pipe`. |
| **Item 3** | [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) | **In-Scope** | Refactor [`RolloutBuffer`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py#L11) to store `[T, N, ...]` steps, implement 2D vectorized GAE calculation, and batch iterator flattening. |
| **Item 4** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **In-Scope** | Modify [`PPO2Agent.get_action`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L65) to support batched inputs without discarding batch dimension. |
| **Item 5** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **In-Scope** | 1. Add global symbol preloading and `SymbolQueue`.<br>2. Initialize `SubprocVecEnv(n_envs=8)`.<br>3. Vectorized rollout loop for $T=500$ steps (4,000 total transitions).<br>4. Multi-worker episode logging & clean termination. |
| **Item 6** | [`test_ppo2_env.py`](file:///home/b0457812963/Mamba3RL/SynapseX/test_ppo2_env.py) | **In-Scope** | Add unit tests verifying `SubprocVecEnv`, batched buffer GAE calculations, and symbol rotation. |
| **Item 7** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **Out-of-Scope** | Neural network architecture (`MambaContinuousActorCritic`) is already fully batched. Untouched. |
| **Item 8** | [`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py) | **Out-of-Scope** | Feature calculation & indicators remain unchanged. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Multiprocessing communication between the main PPO process and 8 worker processes via non-blocking / duplex pipes.
  - Safe memory sharing: Preload all 179 symbols into memory once before multiprocessing fork/spawn.
  - Mathematical correctness of GAE: Per-worker trajectory masking ensuring `done=True` in one worker does not truncate or leak returns into other parallel workers or subsequent episodes.
- **Out-of-Scope (Non-Goals)**:
  - No changes to other algorithms (`DQN`, `A2C`, `A3C`).
  - No changes to observation features or action space boundaries (`Box(-1.0, 1.0)`).

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Update Configuration ([`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml))**
  - [x] Add `NUM_WORKERS: 8`
  - [x] Add `HORIZON_STEPS: 500`
  - [x] Update `N_STEPS: 4000` (meaning total transitions per iteration = $8 \times 500$)
- [x] **Step 2: Preloading Support in [`TrainingEnv`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L315)**
  - [x] Allow `TrainingEnv` to receive an optional `preloaded_data: Optional[Dict[str, Prices]]`.
  - [x] When provided, `_load_data_for_instrument` references `preloaded_data` directly, eliminating duplicate CSV disk reads and redundant memory footprint in child processes.
- [x] **Step 3: Implement Lightweight `SubprocVecEnv` in [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)**
  - [x] Implement `worker_process(remote, parent_remote, env_fn)` responding to commands:
    - `"step"`: executes `env.step(action)`. If `done`, captures terminal info, automatically resets with a new symbol specified or sampled, and packs result.
    - `"reset"`: resets environment with an optional symbol argument.
    - `"close"`: closes child environment and terminates process.
  - [x] Implement `SubprocVecEnv(env_fns)`:
    - Creates $N$ duplex pipes and child processes (using `fork` context on Linux for zero-copy memory access).
    - `step(actions)`: sends actions to all workers, waits for results, stacks `states` and `time_states` into numpy arrays of shape `(N, seq_dim, feat_dim)`.
    - `reset()`: resets all workers, returns batched observations.
    - `close()`: cleanly joins all worker processes.
- [x] **Step 4: Vectorized `RolloutBuffer` in [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py)**
  - [x] Store transitions per time-step across $N$ environments:
    - `states`: `(T, N, seq_dim, feat_dim)`
    - `time_states`: `(T, N, seq_dim, time_feat_dim)`
    - `actions`: `(T, N, action_dim)`
    - `log_probs`: `(T, N)`
    - `rewards`: `(T, N)`
    - `dones`: `(T, N)`
    - `values`: `(T, N)`
  - [x] Vectorized GAE algorithm across `[T, N]` with non-terminal masking.
  - [x] Flatten `[T, N, ...]` to `[T * N, ...]` in `get_batches()`, shuffling across all collected transitions.
- [x] **Step 5: Batched Inference in [`PPO2Agent`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L9)**
  - [x] In `_preprocess_obs(obs)`: If `states.dim() == 2`, unsqueeze batch dimension. If `states.dim() == 3`, maintain batch dimension `(N, seq_dim, feat_dim)`.
  - [x] In `get_action(obs)`: Return `AgentOutput` where `action` is `(N, action_dim)`, `value` is `(N,)`, and `log_prob` is `(N,)`.
- [x] **Step 6: Implement Global Shuffled `SymbolQueue` & Refactor [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)**
  - [x] Implement `SymbolQueue(unique_symbols)`:
    - Maintains a shuffled list of symbols.
    - `get_next_symbol()` pops next symbol; automatically reshuffles and replenishes when empty, guaranteeing uniform coverage.
  - [x] Preload all 179 symbols once in main process before spawning workers.
  - [x] Instantiate `SubprocVecEnv` with 8 workers, passing preloaded data.
  - [x] Main rollout loop:
    - Collect $T = 500$ steps across all 8 environments (total $4,000$ steps per iteration).
    - If worker $i$ finishes an episode (`dones[i] == True`), assign next symbol from `SymbolQueue` and log the completed symbol's episode return.
    - Bootstrap with batched `last_values = agent.get_action(obs).value`.
    - Compute GAE and execute PPO mini-batch updates.
    - Gracefully close `SubprocVecEnv` on `KeyboardInterrupt` or training exit.
- [x] **Step 7: Defensive Verification & Unit Tests**
  - [x] Verify multiprocessing pipe stability, memory consumption (< 6 GB RAM), and GPU VRAM (< 500 MB).
  - [x] Run test suite with `/home/b0457812963/Mamba3RL/bin/python`.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 `SubprocVecEnv` Architecture in [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)
```python
def _worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    info["terminal_observation"] = obs
                    # auto-reset with optional next symbol
                    next_sym = data.get("next_symbol", None) if isinstance(data, dict) else None
                    obs = env.reset(symbol=next_sym)
                remote.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset(symbol=data)
                remote.send(obs)
            elif cmd == "close":
                remote.close()
                break
    except Exception as e:
        remote.send(("error", e))
    finally:
        if hasattr(env, "close"):
            env.close()
```

#### 3.2.2 Vectorized GAE Blueprint in [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py)
```python
def compute_gae(self, last_values: torch.Tensor):
    # self.rewards: (T, N), self.values: (T, N), self.dones: (T, N)
    T, N = self.rewards.shape
    advantages = torch.zeros((T, N), dtype=torch.float32)
    last_gae = torch.zeros(N, dtype=torch.float32)
    last_val = last_values.float()

    for t in reversed(range(T)):
        next_val = last_val if t == T - 1 else self.values[t + 1]
        next_non_terminal = 1.0 - self.dones[t]
        delta = self.rewards[t] + self.gamma * next_val * next_non_terminal - self.values[t]
        last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    self.returns = advantages + self.values
    self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

### 4.2 Acceptance Verification Steps
1. **Unit Test for `SubprocVecEnv` & `SymbolQueue`**:
   - Create a test script verifying that 8 workers launch cleanly, step simultaneously, and return dictionary observations matching the shapes `states: (8, 300, 26)`, `time_states: (8, 300, 14)`.
   - Verify that when a worker finishes an episode, it consumes the next symbol from `SymbolQueue` without repeating until all 179 symbols have been visited.
2. **Unit Test for Vectorized `RolloutBuffer`**:
   - Synthetic trajectory test verifying that `dones[t, worker_idx] == 1.0` isolates GAE calculation for that worker without bleeding reward advantages into the next episode or parallel workers.
3. **End-to-End Runtime & Resource Verification**:
   - Run a 2-iteration live test of `PPO2_rl_train.py` ($2 \times 4000 = 8000$ steps).
   - Check system telemetry:
     - RAM consumption remains stable under ~6 GB.
     - GPU VRAM consumption remains under ~500 MB.
     - FPS throughput exceeds 300+ steps/second.
     - Iteration output correctly logs symbols and rewards without deadlocks or unhandled exceptions.

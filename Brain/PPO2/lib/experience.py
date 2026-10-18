from typing import Dict, List, Optional, Union
from collections import namedtuple
import numpy as np
import torch

Transition = namedtuple(
    "Transition",
    ["state", "action", "logp", "reward", "next_state", "done", "value"],
)


class RolloutBuffer:
    """
    支援向量化多環境 (T x N) 的 PPO Rollout 經驗收集池。
    支援 2D GAE 反向遞推計算與多進程維度獨立截斷，Mini-batch 抽樣時展平為 (T*N) 樣本。
    同時提供 .buffer 與 Transition 結構以確保向後相容性。
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self.states = []
        self.time_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.returns = None
        self.advantages = None

    def store(
        self,
        state: Dict[str, Union[np.ndarray, torch.Tensor]],
        action: Union[np.ndarray, torch.Tensor, float],
        logp: Union[np.ndarray, torch.Tensor, float],
        reward: Union[np.ndarray, float],
        next_state: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        done: Union[np.ndarray, bool] = False,
        value: Union[np.ndarray, torch.Tensor, float] = 0.0,
    ):
        s_data = state["states"]
        if torch.is_tensor(s_data):
            s_data = s_data.cpu().numpy()
        t_data = state["time_states"]
        if torch.is_tensor(t_data):
            t_data = t_data.cpu().numpy()

        act = action.cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        lp = logp.cpu().numpy() if torch.is_tensor(logp) else np.asarray(logp)
        rew = np.asarray(reward, dtype=np.float32)
        dn = np.asarray(done, dtype=bool)
        val = (
            value.cpu().numpy()
            if torch.is_tensor(value)
            else np.asarray(value, dtype=np.float32)
        )

        # 支援單環境輸入：如果維度為 scalar 或 2D state，自動擴展至 N=1 維度
        if s_data.ndim == 2:
            s_data = np.expand_dims(s_data, 0)
            t_data = np.expand_dims(t_data, 0)
            act = np.expand_dims(act, 0)
            lp = np.reshape(lp, (1,))
            rew = np.reshape(rew, (1,))
            dn = np.reshape(dn, (1,))
            val = np.reshape(val, (1,))

        if act.ndim == 1:
            act = np.expand_dims(act, -1)

        self.states.append(s_data)
        self.time_states.append(t_data)
        self.actions.append(act)
        self.log_probs.append(lp)
        self.rewards.append(rew)
        self.dones.append(dn)
        self.values.append(val)

    def compute_gae(
        self,
        last_values: Optional[Union[torch.Tensor, np.ndarray, float]] = None,
        last_value: Optional[Union[torch.Tensor, np.ndarray, float]] = None,
    ):
        """
        計算向量化二維廣義優勢估計 (GAE, Generalized Advantage Estimation)。
        支援單環境與 N 個環境並行軌跡，利用 dones[t] 遮罩防止跨 Episode 優勢洩漏。
        同時相容 last_value (單環境) 與 last_values (向量化環境) 參數名。
        """
        if last_values is None:
            last_values = last_value if last_value is not None else 0.0

        T = len(self.rewards)
        if T == 0:
            return self.buffer

        # 堆疊為 shape (T, N)
        rewards = torch.as_tensor(np.stack(self.rewards, axis=0), dtype=torch.float32)
        values = torch.as_tensor(np.stack(self.values, axis=0), dtype=torch.float32)
        dones = torch.as_tensor(np.stack(self.dones, axis=0), dtype=torch.float32)

        N = rewards.shape[1]
        if torch.is_tensor(last_values):
            last_val = last_values.detach().cpu().float().view(-1)
        else:
            last_val = torch.as_tensor(last_values, dtype=torch.float32).view(-1)
        if last_val.numel() == 1 and N > 1:
            last_val = last_val.expand(N)

        advantages = torch.zeros((T, N), dtype=torch.float32)
        last_gae = torch.zeros(N, dtype=torch.float32)

        for step in reversed(range(T)):
            next_val = last_val if step == T - 1 else values[step + 1]
            next_non_terminal = 1.0 - dones[step]
            delta = (
                rewards[step]
                + self.gamma * next_val * next_non_terminal
                - values[step]
            )
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            advantages[step] = last_gae

        returns = advantages + values

        # 展平為 (T * N,)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        # 儲存未標準化的優勢與回報（標準化改於 Mini-Batch 層級執行，對齊 OpenAI Baselines / CleanRL）
        self.advantages = flat_advantages
        self.returns = flat_returns
        return self.buffer

    @property
    def buffer(self):
        # 向後相容屬性：將內部張量還原為 Transition namedtuple 串列
        trans_list = []
        T = len(self.rewards)
        if T == 0:
            return trans_list
        returns = (
            self.returns
            if self.returns is not None
            else [0.0] * (T * len(self.rewards[0]))
        )
        advantages = (
            self.advantages
            if self.advantages is not None
            else [0.0] * (T * len(self.rewards[0]))
        )
        idx = 0
        for t in range(T):
            N = len(self.rewards[t])
            for i in range(N):
                st = {
                    "states": self.states[t][i],
                    "time_states": self.time_states[t][i],
                }
                ret = (
                    float(returns[idx])
                    if idx < len(returns)
                    else float(self.rewards[t][i])
                )
                adv = (
                    float(advantages[idx])
                    if idx < len(advantages)
                    else float(self.values[t][i])
                )
                trans_list.append(
                    Transition(
                        state=st,
                        action=self.actions[t][i],
                        logp=self.log_probs[t][i],
                        reward=ret,
                        next_state=None,
                        done=bool(self.dones[t][i]),
                        value=adv,
                    )
                )
                idx += 1
        return trans_list

    def get_batches(self, batch_size: int = 64, shuffle: bool = True):
        if len(self.states) == 0 or self.advantages is None:
            return

        # 堆疊為 (T * N, ...)
        states = torch.as_tensor(
            np.concatenate(self.states, axis=0), dtype=torch.float32
        )
        time_states = torch.as_tensor(
            np.concatenate(self.time_states, axis=0), dtype=torch.float32
        )
        actions = torch.as_tensor(
            np.concatenate(self.actions, axis=0), dtype=torch.float32
        )
        old_log_probs = torch.as_tensor(
            np.concatenate(self.log_probs, axis=0), dtype=torch.float32
        ).reshape(-1)
        old_values = torch.as_tensor(
            np.concatenate(self.values, axis=0), dtype=torch.float32
        ).reshape(-1)

        dataset_size = len(self.advantages)
        assert dataset_size % batch_size == 0, (
            f"Rollout dataset size ({dataset_size}) must be evenly divisible by batch_size ({batch_size}) "
            f"to prevent gradient scale imbalance from fractional trailing batches."
        )
        indices = np.arange(dataset_size)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, dataset_size, batch_size):
            b_idx = indices[start_idx : start_idx + batch_size]
            yield (
                states[b_idx],
                time_states[b_idx],
                actions[b_idx],
                old_log_probs[b_idx],
                self.returns[b_idx],
                self.advantages[b_idx],
                old_values[b_idx],
            )

    def clear(self):
        self.states.clear()
        self.time_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.returns = None
        self.advantages = None

    def __len__(self):
        return sum(len(s) for s in self.states)
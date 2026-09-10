import numpy as np
import torch
from collections import namedtuple

Transition = namedtuple(
    'Transition',
    ['state', 'action', 'logp', 'reward', 'next_state', 'done', 'value'],
)


class RolloutBuffer:
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self.buffer = []

    def store(self, *args):
        self.buffer.append(Transition(*args))

    def compute_gae(self, last_value):
        rewards, values, dones = [], [], []
        for t in self.buffer:
            rewards.append(float(t.reward))
            values.append(
                t.value.item() if torch.is_tensor(t.value) else float(t.value)
            )
            dones.append(bool(t.done))

        last_val = (
            last_value.item() if torch.is_tensor(last_value) else float(last_value)
        )
        values = values + [last_val]
        gae = 0.0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1.0 - float(dones[step]))
                - values[step]
            )
            gae = delta + self.gamma * self.lam * (1.0 - float(dones[step])) * gae
            returns.insert(0, gae + values[step])

        advantages = np.array(returns) - np.array(values[:-1])
        for idx, tr in enumerate(self.buffer):
            self.buffer[idx] = tr._replace(reward=returns[idx], value=advantages[idx])
        return self.buffer

    def get_batches(self, batch_size: int = 64, shuffle: bool = True):
        if len(self.buffer) == 0:
            return

        states = torch.stack(
            [torch.as_tensor(t.state["states"], dtype=torch.float32) for t in self.buffer]
        )
        time_states = torch.stack(
            [
                torch.as_tensor(t.state["time_states"], dtype=torch.float32)
                for t in self.buffer
            ]
        )
        actions = torch.stack(
            [torch.as_tensor(t.action, dtype=torch.float32) for t in self.buffer]
        )
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        old_log_probs = torch.tensor(
            [
                t.logp.item() if torch.is_tensor(t.logp) else float(t.logp)
                for t in self.buffer
            ],
            dtype=torch.float32,
        )
        returns = torch.tensor([t.reward for t in self.buffer], dtype=torch.float32)
        advantages = torch.tensor([t.value for t in self.buffer], dtype=torch.float32)

        # Advantage 標準化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(self.buffer)
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
                returns[b_idx],
                advantages[b_idx],
            )

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
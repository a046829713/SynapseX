from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# 假設這些模組你已經有了，保持不變
from nested_learning.hope.block import HOPEBlock, HOPEBlockConfig
from nested_learning.levels import LevelSpec


@dataclass
class DQNConfig:
    state_dim: int        # 修改: 輸入狀態的維度 (例如: 遊戲畫面的特徵長度 或 感測器數據量)
    action_dim: int       # 修改: 動作空間的大小 (Q-values 的數量)
    dim: int
    num_layers: int
    heads: int
    titan_level: LevelSpec
    cms_levels: Sequence[LevelSpec]
    optimizers: Dict[str, dict] | None = None
    teach_scale: float = 1.0
    teach_clip: float = 0.0
    teach_schedule: Dict[str, float] | None = None
    gradient_checkpointing: bool = False
    surprise_threshold: float | None = None

class HOPEDQN(nn.Module):
    def __init__(self, config: DQNConfig):
        super().__init__()
        self.config = config
        
        # 修改 1: 輸入層
        # RL 的輸入通常是連續數值 (Float)，而非離散 Token (Int)
        # 我們使用一個 Linear 層將 State 投影到模型的內部維度 dim
        self.state_encoder = nn.Linear(config.state_dim, config.dim)
        
        self.base_teach_scale = config.teach_scale
        self.base_teach_clip = config.teach_clip
        self._runtime_teach_scale = config.teach_scale
        self._runtime_teach_clip = config.teach_clip
        self.gradient_checkpointing = config.gradient_checkpointing
        self._surprise_threshold = config.surprise_threshold
        self._allowed_update_levels: set[str] | None = None
        
        block_config = HOPEBlockConfig(
            dim=config.dim,
            heads=config.heads,
            titan_level=config.titan_level,
            cms_levels=config.cms_levels,
            optimizer_configs=config.optimizers or {},
        )
        self.blocks = nn.ModuleList([HOPEBlock(block_config) for _ in range(config.num_layers)])
        
        self.norm = nn.LayerNorm(config.dim)
        
        # 修改 2: 輸出層 (Q-Head)
        # 輸出 Q(s, a)，大小為 action_dim
        # 移除了 bias=False 的限制，RL 中通常需要 bias
        self.action_head = nn.Linear(config.dim, config.action_dim)
        
        # 注意: 移除了原本 LM 的 Weight Tying (self.lm_head.weight = self.embed.weight)
        # 因為 State 和 Action 處於不同的空間
        
        self._latest_update_metrics: Dict[str, float] = {}
        self.set_surprise_threshold(self._surprise_threshold)

    def set_teach_runtime(self, *, scale: float | None = None, clip: float | None = None) -> None:
        if scale is not None:
            self._runtime_teach_scale = scale
        if clip is not None:
            self._runtime_teach_clip = clip

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self._surprise_threshold = threshold
        for block in self.blocks:
            block.set_surprise_threshold(threshold)

    def get_surprise_threshold(self) -> float | None:
        return self._surprise_threshold

    def set_allowed_update_levels(self, levels: set[str] | None) -> None:
        self._allowed_update_levels = levels.copy() if levels is not None else None
        for block in self.blocks:
            block.set_allowed_levels(self._allowed_update_levels)

    def get_allowed_update_levels(self) -> set[str] | None:
        return None if self._allowed_update_levels is None else self._allowed_update_levels.copy()

    def forward(
        self,
        state: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            state: 形狀為 (Batch_Size, State_Dim) 的張量
            teach_signal: 可選的教學信號，用於 HOPE 架構的內部調整
        Returns:
            q_values: 形狀為 (Batch_Size, Action_Dim) 的張量
        """
        
        # 1. Encode State
        # state shape: [Batch, State_Dim] -> x shape: [Batch, Dim]
        x = self.state_encoder(state)

        # 重要: 如果 HOPEBlock 內部包含 Attention 機制，它通常預期輸入為 (Batch, Sequence, Dim)
        # 如果 DQN 處理的是單一時間步，我們需要增加一個虛擬的序列維度
        # 如果你的 HOPEBlock 是純 MLP 結構，這行可能不需要，但在 Transformer 架構下通常需要。
        if x.dim() == 2:
            x = x.unsqueeze(1) # [Batch, 1, Dim]

        surprise_value: float | None = None
        
        if teach_signal is not None:
            # 確保 teach_signal 維度匹配，如果它也是 (Batch, Dim)
            if teach_signal.dim() == 2 and x.dim() == 3:
                 teach_signal = teach_signal.unsqueeze(1)
            surprise_value = float(teach_signal.norm(dim=-1).mean().item())

        for block in self.blocks:
            scaled_signal = None
            if teach_signal is not None:
                scaled_signal = teach_signal * self._runtime_teach_scale
                if self._runtime_teach_clip > 0:
                    norm = scaled_signal.norm(dim=-1, keepdim=True)
                    scale = torch.clamp(norm / self._runtime_teach_clip, min=1.0)
                    scaled_signal = scaled_signal / scale

            # Block Call
            block_call = lambda hidden, blk=block, sig=scaled_signal: blk(
                hidden,
                teach_signal=sig,
                surprise_value=surprise_value,
            )

            if self.training and self.gradient_checkpointing:
                # 我在猜想可能是後面為了要減少記憶體作者所做的改寫
                x = checkpoint(block_call, x, use_reentrant=False)
            else:
                x = block_call(x)

        # 輸出前標準化
        x = self.norm(x)
        
        # 如果我們之前增加了一個序列維度，現在把它拿掉
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1) # 回到 [Batch, Dim]

        # 計算 Q-Values
        q_values = self.action_head(x) # [Batch, Action_Dim]

        if teach_signal is not None:
            self._latest_update_metrics = self._gather_block_stats()

        return q_values

    def _gather_block_stats(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for idx, block in enumerate(self.blocks):
            if hasattr(block, "pop_update_stats"):
                stats = block.pop_update_stats()
                for level_name, payload in stats.items():
                    prefix = f"layer{idx}.{level_name}"
                    for key, value in payload.items():
                        metrics[f"{prefix}.{key}"] = value
        return metrics

    def pop_update_metrics(self) -> Dict[str, float]:
        metrics = self._latest_update_metrics
        self._latest_update_metrics = {}
        return metrics
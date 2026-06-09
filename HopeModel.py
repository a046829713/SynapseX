from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# 假設這些模組你已經有了，保持不變
from nested_learning.hope.block import HOPEBlock, HOPEBlockConfig
from nested_learning.levels import LevelSpec
from Brain.Common.model_components import SineActivation
import time
from Brain.Common.dain import DAIN_Layer

@dataclass
class DQNConfig:
    state_dim: int        # 修改: 輸入狀態的維度 (例如: 遊戲畫面的特徵長度 或 感測器數據量)
    action_dim: int       # 修改: 動作空間的大小 (Q-values 的數量)
    time_dim:int
    time_features_out:int
    mode:str
    hidden_size: int
    seq_dim :int
    dropout :float
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
        self.time_embedding = SineActivation(in_features=config.time_dim, out_features=config.time_features_out)
        self.time_emb_projection = nn.Linear(config.time_features_out, config.hidden_size)

        self.dean = DAIN_Layer(mode=config.mode, input_dim=config.state_dim) # DAIN 只處理市場數據

        self.market_embedding = nn.Linear(config.state_dim, config.hidden_size)
        


        # 門控層: 學習如何結合兩種資訊
        self.gate_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )

        # 最終的特徵轉換
        self.feature_embedding = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU()
        )


        # 狀態值網絡
        self.fc_val = nn.Sequential(
            nn.Linear(config.seq_dim * config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 1)
        )

        # 優勢網絡
        self.fc_adv = nn.Sequential(
            nn.Linear(config.seq_dim * config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.action_dim)
        )




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
        src: torch.Tensor,
        time_state: torch.Tensor,
        teach_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            state: 形狀為 (Batch_Size, State_Dim) 的張量
            teach_signal: 可選的教學信號，用於 HOPE 架構的內部調整
        
        
        
        Returns:
            q_values: 形狀為 (Batch_Size, Action_Dim) 的張量
        """
        
        time_emb = self.time_embedding(time_state)
        time_emb_proj = self.time_emb_projection(time_emb) # [B, L, hidden_size]
        
        # 市場數據流
        market_data = src.transpose(1, 2)
        market_data = self.dean(market_data)
        market_data = market_data.transpose(1, 2)
        market_emb = self.market_embedding(market_data) # [B, L, hidden_size]

        # 計算門控值
        gate = self.gate_layer(torch.cat([market_emb, time_emb_proj], dim=-1))
        
        # 融合特徵
        fused_emb = gate * market_emb + (1 - gate) * time_emb_proj
        x = self.feature_embedding(fused_emb)



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

        x = x.view(x.size(0), -1)

        value = self.fc_val(x)       # [B, 1]
        advantage = self.fc_adv(x)   # [B, num_actions]

        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
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
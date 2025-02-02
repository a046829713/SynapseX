# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import copy
import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import time
# this code if from mamba_ssm.models import mixer_seq_simple
def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    """
    簡要說明:
      - 依據給定的參數生成一個 Block，裡面可能包含了 SSM (Mamba/Mamba2) 或 Attention (MHA)。
      - 以及一個 MLP (預設為 GatedMLP) 與對應的層級正規化 (LayerNorm 或 RMSNorm)。
      - Block 物件內部會把上面的組件包裝起來，並在 forward 時依序呼叫。

    主要流程:
      1. 判斷該層是否使用注意力層 (MHA) 還是 SSM (Mamba)。
      2. 產生對應的混合器 (mixer_cls)。
      3. 產生對應的正規化層 (norm_cls)。
      4. 產生 MLP (如 d_intermediate > 0) 或 Identity。
      5. 將上述元件包裝成 Block。
    """
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}


    # 判斷是否在這一層用 Attention
    if layer_idx not in attn_layer_idx:
        # 若不用 Attention，就用 Mamba1 or Mamba2 (SSM)
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")

        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        # 如果這一層需要 Attention，就建立一個 MHA (Multi-Head Attention)
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)

    # 決定要用哪種 Norm 函式（LayerNorm 或 RMSNorm）
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    # 若 d_intermediate 為 0，則不需要 MLP，直接用 Identity
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )

    # 建立一個 Block，並把上面定義好的 mixer, mlp, norm 全部放進去
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    """
    用於自動初始化線性層、Embedding 等等的權重。

    rescale_prenorm_residual = True 時會根據 GPT-2 的初始化策略，
    對特定的權重做 1/sqrt(2*n_layer) 的縮放。
    """
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    # GPT-2 類似的縮放初始化
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    """
        核心說明:
        - MixerModel 是一個將 Token Embedding、若干個 Block (可能是SSM或Attention) 與最終的
            LayerNorm (或 RMSNorm) 組合而成的模型骨幹 (Backbone)。
        - 在 forward 時，會先把輸入 tokens 轉成向量 (embedding)，然後經過所有 Block，
            最後再做一次 Norm。
        - 如果使用 fused_add_norm，則會把 (殘差 + LayerNorm) 的操作用 Triton kernel 做融合，提升效能。
    """

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,        
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        """
        參數簡要:
          - d_model: 每個 token embedding 的維度，也是 Block 的主維度。
          - n_layer: 堆疊多少個 Block。
          - d_intermediate: MLP 中間層維度，如為0表示無MLP。          
          - ssm_cfg, attn_cfg: 關於 SSM/Attention 的自定義參數。
          - attn_layer_idx: 哪些 layer_idx 應該用 Attention，而不是用 SSM。
          - norm_epsilon: Norm 層的 eps。
          - rms_norm: 是否採用 RMSNorm，否則就用 LayerNorm。
          - fused_add_norm: 是否使用 Triton kernel 融合殘差和正規化。
          - residual_in_fp32: 殘差計算是否保存在 FP32（可減少精度誤差）。
          - device, dtype: 指定要用的設備與資料型態。
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # 2. 決定是否使用 Triton 的 fused add + norm 來加速
        self.fused_add_norm = fused_add_norm

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        # 3. 建立 n_layer 個 Block (SSM/MHA + MLP + Norm)
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        # 4. 最後再加一個 Norm 層（如 GPT 類模型會有一個最後的 LayerNorm）
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        # 5. 給所有參數做初始化
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,
            )
        )


    def forward(self, x, inference_params=None, **mixer_kwargs):
        """
            前向傳播流程:
            1. 根據 input_ids (batch_size, seq_len) 取得 Token Embedding。
            2. 將 embedding 丟入每個 Block，並維護 residual（殘差）。
            3. 最後把最終的 hidden_states 做一次 Norm 後輸出。
            
            參數:
            - input_ids: (batch_size, seq_len) 的整數張量。
            - inference_params: 推理相關的 cache 參數 (如 KV cache)。
            - mixer_kwargs: 傳給 mixer 的額外參數（例如如果是注意力，可能有 attn_mask 等）。
            回傳:
            - (batch_size, seq_len, d_model) 的張量。
        """
        # 1. 取得 embedding
        hidden_states = x
        residual = None

        # 2. 逐層 forward
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, 
                residual, 
                inference_params=inference_params, 
                **mixer_kwargs
            )

        # 3. 最終再做一次 Norm
        if not self.fused_add_norm:
            # 如果不用 fused kernel，就手動把 residual 加回 hidden_states
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # 如果用 fused kernel，就呼叫 layer_norm_fn 做殘差和 norm 的融合
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states
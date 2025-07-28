# Copyright (c) 2023, Albert Gu, Tri Dao.
import math
from functools import partial
import copy
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
#
# 步驟 1: 建立新的 MoELayer
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
# from mamba_ssm.modules.block import Block
import time


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim) # Mamba or MHA
        print(self.mixer)
        # mlp_cls 可以是 GatedMLP 也可以是 MoELayer
        self.mlp = mlp_cls() if mlp_cls is not nn.Identity else nn.Identity()
        print("Block build success.")

        # <<< MODIFIED: 只有當 mlp 不是 Identity 時才創建 norm2 >>>
        if not isinstance(self.mlp, nn.Identity):
            self.norm2 = norm_cls(dim)
        else:
            self.norm2 = None # 如果沒有 mlp，就不需要 norm2
        
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        # <<< MODIFIED: 初始化 aux_loss 為 None >>>
        aux_loss = None

        # --- 第一個子模組 (Mixer: Mamba/MHA) ---
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states, self.norm.weight, self.norm.bias, residual=residual, prenorm=True,
                residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        # --- 第二個子模組 (MLP: GatedMLP/MoE) ---
        # <<< MODIFIED: 只有當 self.mlp 和 self.norm2 存在時才執行 >>>
        if self.mlp is not None and self.norm2 is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states, self.norm2.weight, self.norm2.bias, residual=residual, prenorm=True,
                    residual_in_fp32=self.residual_in_fp32, eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )

            # <<< MODIFIED: 處理 MLP 的輸出 >>>
            # 檢查 self.mlp 的輸出，如果是 tuple，則表示是 MoE 層
            mlp_out = self.mlp(hidden_states)
            if isinstance(mlp_out, tuple):
                hidden_states, aux_loss = mlp_out
            else:
                hidden_states = mlp_out

        # <<< MODIFIED: 回傳 hidden_states, residual 和 aux_loss >>>
        return hidden_states, residual, aux_loss

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)







class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        gated_mlp_cls: nn.Module, # 將 GatedMLP 作為參數傳入
        **gated_mlp_kwargs
    ):
        """
        一個 Mixture of Experts 層。

        參數:
          - d_model: 輸入和輸出的維度。
          - num_experts: 專家的總數量。
          - top_k: 每次為每個 token 選擇 top_k 個專家。
          - gated_mlp_cls: 用來創建專家的類別 (例如 GatedMLP)。
          - gated_mlp_kwargs: 創建專家時需要傳入的參數。
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # 1. 建立專家網路
        # 每個專家都是一個 GatedMLP 實例
        self.experts = nn.ModuleList([
            gated_mlp_cls(in_features=d_model, **gated_mlp_kwargs) for _ in range(num_experts)
        ])

        # 2. 建立門控網路
        # 一個簡單的線性層，輸出每個專家的 logits
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # 3. 負載平衡損失相關的噪音參數 (可選但建議)
        self.noise_epsilon = 1e-2
        self.register_buffer('mean', torch.tensor([0.0]))
        self.register_buffer('std', torch.tensor([1.0]))


    def _compute_load_balancing_loss(self, gate_logits, num_tokens):
        """
        計算輔助的負載平衡損失 (參考 Switch Transformers 和 Mixtral 的實現)。
        這個損失有助於確保所有專家都得到充分的訓練。
        """
        # gate_logits: [num_tokens, num_experts]
        top_logits, top_indices = gate_logits.topk(self.top_k, dim=1)
        
        # 將 logits 轉換為機率分佈
        gates_softmax = F.softmax(gate_logits, dim=1) # [num_tokens, num_experts]
        
        # 計算每個專家被選擇的頻率 (f_i)
        # top_indices: [num_tokens, top_k]
        # F.one_hot 會產生 [num_tokens, top_k, num_experts] 的張量
        # .sum(dim=1) 後得到 [num_tokens, num_experts]，表示每個 token 選擇了哪些專家
        temp_mask = F.one_hot(top_indices, self.num_experts).sum(dim=1)
        
        # f_i: 每個專家處理的 token 比例
        f_i = temp_mask.float().mean(dim=0) # [num_experts]
        
        # P_i: 門控網路輸出權重的平均值
        P_i = gates_softmax.mean(dim=0) # [num_experts]
        
        # Loss = (f_i * P_i).sum()
        # 乘以專家數量的平方是一個常用的縮放因子
        loss = (f_i * P_i).sum() * (self.num_experts ** 2)
        return loss


    def forward(self, x: torch.Tensor):
        """
        前向傳播。

        參數:
          - x: 輸入張量，形狀為 [batch_size, seq_len, d_model]

        回傳:
          - final_output: 處理後的輸出張量，形狀同 x。
          - aux_loss: 輔助的負載平衡損失。
        """
        batch_size, seq_len, d_model = x.shape

        # 將輸入 reshape 成 token 列表，方便處理
        x_flat = x.view(-1, d_model) # [B * L, D]
        num_tokens = x_flat.shape[0]

        
        # torch.Size([300, 8])
        gate_logits = self.gate(x_flat)
        
        # 1. 透過門控網路計算 logits
        # 增加一些噪音可以改善負載平衡 (Jittering)
        if self.training:
            normal_dist = Normal(self.mean, self.std)
            noise = normal_dist.sample(gate_logits.shape).squeeze() * self.noise_epsilon
            gate_logits = gate_logits + noise

            
        # 2. 選擇 top-k 的專家並計算權重
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=1)

        top_k_weights = F.softmax(top_k_logits, dim=1, dtype=torch.float).to(x.dtype) # [B * L, top_k]
        
        # 3. 計算輔助損失
        aux_loss = self._compute_load_balancing_loss(gate_logits, num_tokens)

        # 4. 將 token 分配給專家並計算輸出
        final_output = torch.zeros_like(x_flat)
        # 創建一個扁平化的索引，方便後面用 `scatter_add_`
        flat_top_k_indices = top_k_indices.flatten() # [B * L * top_k]

        # 將輸入 x_flat 擴展，以匹配每個 token 的 top_k 個選擇
        # x_flat -> [B*L, D]
        # top_k_weights -> [B*L, top_k]
        # y -> [B*L, top_k, D]
        y = (x_flat.unsqueeze(1) * top_k_weights.unsqueeze(2)).view(-1, d_model)

        # 初始化一個 expert_outputs 張量
        expert_outputs = torch.zeros_like(y)
        
        # 使用 for 循環遍歷專家 (雖然效率不高，但易於理解)
        # 在實際應用中，會使用更高效的 dispatch/combine 操作
        for i in range(self.num_experts):
            # 找到所有應該由第 i 個專家處理的 token
            mask = (flat_top_k_indices == i)
            if mask.any():
                # 將這些 token 餵給專家
                expert_input = y[mask]
                expert_outputs[mask] = self.experts[i](expert_input)

        # 將所有專家的輸出加總起來
        # y -> [B*L*top_k, D]
        # final_output -> [B*L, D]
        final_output = final_output.scatter_add_(0, top_k_indices.view(-1, 1).expand(-1, d_model), expert_outputs)

        return final_output.view(batch_size, seq_len, d_model), aux_loss








class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.silu,
        bias=False,
        multiple_of=128,
        dropout=None,
        device=None,
        dtype=None,
    ):
        """
            Copyright (c) 2024, Tri Dao, Albert Gu.

            change auther : Louis.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        if hasattr(self, 'dropout'):
            y = self.dropout(y)  # 主路径 dropout
            gate = self.dropout(gate)  # 门控路径 dropout
        y = y * self.activation(gate)
        if hasattr(self, 'dropout'):
            y = self.dropout(y)  # 激活后 dropout
        y = self.fc2(y)
        return y




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
    dropout=None,
    moe_cfg=None,
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
    
    print("開始創建create_block")
    # 若 d_intermediate 為 0，則不需要 MLP，直接用 Identity
    if d_intermediate == 0:
        mlp_cls = nn.Identity    
    elif moe_cfg and moe_cfg.get("num_experts", 0) > 0:
        # 如果 moe_cfg 存在且指定了專家數量，則使用 MoELayer
        print(f"Layer {layer_idx}: Using MoE with {moe_cfg['num_experts']} experts.")
        print("開始創建MoELayer1")
        # 建立 MoELayer
        mlp_cls = partial(
            MoELayer,
            d_model=d_model,
            num_experts=moe_cfg['num_experts'],
            top_k=moe_cfg.get('top_k', 2), # top_k 預設為 2
            gated_mlp_cls=GatedMLP,
            hidden_features=d_intermediate,
            out_features=d_model,
            dropout = dropout,
            **factory_kwargs
        )
        print("開始創建MoELayer2")
    else:
        
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, dropout = dropout, **factory_kwargs
        )


    print(mlp_cls)
    print("開始創建Block")
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
        fused_add_norm=True,
        residual_in_fp32=False,
        moe_cfg=None,
        device=None,
        dtype=None,
        dropout=None
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
        
        print("開始創建layer1:")
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
                    dropout=dropout,
                    moe_cfg=moe_cfg,
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

        # 【修改點】
        # 初始化一個列表來收集所有層的輔助損失
        aux_losses = []


        # 2. 逐層 forward
        for layer in self.layers:
            # Block 的 forward 需要被修改以回傳 aux_loss
            # 這一步驟的修改取決於你的 Block 類別的具體實現
            # 假設 Block 的 forward 現在回傳 (hidden_states, residual, aux_loss)
            hidden_states, residual, aux_loss = layer(
                hidden_states, 
                residual, 
                inference_params=inference_params, 
                **mixer_kwargs
            )
            if aux_loss is not None:
                aux_losses.append(aux_loss)

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

        total_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        return hidden_states, total_aux_loss
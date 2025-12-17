
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder
from Brain.Common.transformer_tool import TransformerEncoderLayer, PositionalEncoding
from Brain.Common.dain import DAIN_Layer
from Brain.Common.model_components import SineActivation
from einops import rearrange
from Brain.Common.ssm_tool import MixerModel,GatedMLP
from torch.nn import functional as F
from typing import Optional
import time
import soft_moe_pytorch

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

except Exception as e:
    print(e)
    print("causal_conv1d not found.")
    print("please check the package.")


# class TransformerDuelingModel(nn.Module):
#     def __init__(self,
#                  d_model: int,
#                  nhead: int,
#                  d_hid: int,
#                  nlayers: int,
#                  num_actions: int,
#                  hidden_size: int,
#                  seq_dim: int = 300,
#                  dropout: float = 0.5,
#                  batch_first=True,
#                  mode='full'):
#         """
#             原本EncoderLayer 是使用官方的，後面因為訓練上難以收斂
#             故重新在製作一次屬於自己的TransformerEncoderLayer
#         """

#         super().__init__()
#         self.batch_first = batch_first
#         self.pos_encoder = PositionalEncoding(hidden_size, dropout)

#         encoder_layers = TransformerEncoderLayer(
#             hidden_size, nhead, d_hid, dropout, batch_first=self.batch_first)

#         self.transformer_encoder = TransformerEncoder(
#             encoder_layers, nlayers, norm=nn.LayerNorm(hidden_size), enable_nested_tensor=False)

#         self.dean = DAIN_Layer(mode=mode,input_dim=d_model)

#         # 狀態值網絡
#         self.fc_val = nn.Sequential(
#             nn.Linear(seq_dim * hidden_size // 8, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, 1)
#         )

#         # 優勢網絡
#         self.fc_adv = nn.Sequential(
#             nn.Linear(seq_dim * hidden_size // 8, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_actions)
#         )

#         self.linear = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 2, hidden_size // 4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 4, hidden_size // 8)
#         )

#         # 將資料映射
#         self.embedding = nn.Sequential(
#             nn.Linear(d_model, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size)
#         )

#         self.embed_ln = nn.LayerNorm(hidden_size)  # 層歸一化

#         # 初始化權重
#         # self.apply(self._init_weights)
        

#     # def _init_weights(self, m):
#     #     if isinstance(m, nn.Linear):
#     #         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He初始化
#     #         if m.bias is not None:
#     #             nn.init.constant_(m.bias, 0)

#     def forward(self, src: Tensor) -> Tensor:
#         """
#         Arguments:
#             src: Tensor, shape ``[batch_size, seq_len, d_model]``

#         Returns:
#             output Tensor of shape ``[batch_size, num_actions]``

#         """
#         # 
#         src = src.transpose(1, 2)
#         src = self.dean(src)
#         src = src.transpose(1, 2)

#         src = self.embedding(src)

#         # src = torch.Size([1, 300, 6])
#         if self.batch_first:
#             src = self.pos_encoder(src.transpose(0, 1))
#         else:
#             src = self.pos_encoder(src)

#         src = self.embed_ln(src.transpose(0, 1))

#         if self.batch_first:
#             output = self.transformer_encoder(src)
#         else:
#             output = self.transformer_encoder(src.transpose(0, 1))

#         # output = torch.Size([1, 300, 6])
#         x = self.linear(output)
#         x = x.view(x.size(0), -1)

#         # 狀態值和優勢值
#         value = self.fc_val(x)
#         advantage = self.fc_adv(x)

#         q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
#         # 計算最終的Q值

#         return q_values



class COT_TransformerDuelingModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 num_actions: int,
                 hidden_size: int,
                 seq_dim: int = 300,
                 dropout: float = 0.5,
                 batch_first=True,
                 mode='full',
                 num_iterations=3):

        super().__init__()
        self.batch_first = batch_first
        self.num_iterations = num_iterations
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layers =TransformerEncoderLayer(
            hidden_size, nhead, d_hid, dropout, batch_first=self.batch_first)

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers, norm=nn.LayerNorm(hidden_size), enable_nested_tensor=False)

        self.dean = DAIN_Layer(mode=mode, input_dim=d_model)

        # 狀態值網絡
        self.fc_val = nn.Sequential(
            nn.Linear(seq_dim * hidden_size // 8, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # 優勢網絡
        self.fc_adv = nn.Sequential(
            nn.Linear(seq_dim * hidden_size // 8, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions)
        )

        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, hidden_size // 8)
        )

        # 將資料映射至hidden_size維度
        self.embedding = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.embed_ln = nn.LayerNorm(hidden_size)

        # 用於將多次迭代的輸出混合的LayerNorm
        self.iteration_ln = nn.LayerNorm(hidden_size)

        # Gating 機制
        # 假設透過一個簡單的線性層將 [src_embed, output] concat後得到gate值
        self.gate = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, src: Tensor) -> Tensor:
        # src: [batch_size,seq_len,d_model]
        src = src.transpose(1, 2)
        src = self.dean(src) # [B, seq_len, d_model]
        src = src.transpose(1, 2)

        # 初始embedding
        src_embed = self.embedding(src)  # [B, seq_len, hidden_size]

        # 位置編碼
        if self.batch_first:
            src_embed = self.pos_encoder(src_embed.transpose(0, 1))
            src_embed = src_embed.transpose(0, 1)
        else:
            src_embed = self.pos_encoder(src_embed)

        # embedding層的LN
        src_embed = self.embed_ln(src_embed)
        
        # 開始多次迭代 (chain-of-thought)
        for _ in range(self.num_iterations):            
            if self.batch_first:
                output = self.transformer_encoder(src_embed) # [B, seq_len, hidden_size]
            else:
                output = self.transformer_encoder(src_embed.transpose(0, 1)).transpose(0, 1)
            
            # Gating combine
            # concat後沿最後維度拼接: [B, seq_len, hidden_size*2]
            combined = torch.cat([src_embed, output], dim=-1)

            g = self.gate(combined)  # [B, seq_len, hidden_size]
            # 使用gate決定新狀態
            src_embed = g * output + (1 - g) * src_embed
            src_embed = self.iteration_ln(src_embed)
        

        x = self.linear(src_embed)  
        x = x.view(x.size(0), -1)

        # 狀態值和優勢值
        value = self.fc_val(x)       # [B, 1]
        advantage = self.fc_adv(x)   # [B, num_actions]

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    















class TransformerDuelingModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 num_actions: int,
                 hidden_size: int,
                 seq_dim: int = 300,
                 dropout: float = 0.5,
                 batch_first=True,
                 mode='full',
                 num_iterations=1):

        super().__init__()
        self.batch_first = batch_first
        self.num_iterations = num_iterations
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layers =TransformerEncoderLayer(
            hidden_size, nhead, d_hid, dropout, batch_first=self.batch_first)

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers, norm=nn.LayerNorm(hidden_size), enable_nested_tensor=False)

        self.dean = DAIN_Layer(mode=mode, input_dim=d_model)

        # 狀態值網絡
        self.fc_val = nn.Sequential(
            nn.Linear(seq_dim * hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # 優勢網絡
        self.fc_adv = nn.Sequential(
            nn.Linear(seq_dim * hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions)
        )


        # 將資料映射至hidden_size維度
        self.feature_embedding = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.embed_ln = nn.LayerNorm(hidden_size)

        # 用於將多次迭代的輸出混合的LayerNorm
        self.iteration_ln = nn.LayerNorm(hidden_size)

        self.gatemlp = GatedMLP(in_features=hidden_size, hidden_features=1, out_features=hidden_size, dropout = dropout)
        

    def forward(self, src: Tensor) -> Tensor:
        # src: [batch_size, seq_len, d_model]

        # 根據實測 rearrange 比較慢一些
        # src = rearrange(src,'b l d -> b d l')        
        src = src.transpose(1, 2)        
        src = self.dean(src) # [B, seq_len, d_model]
        src = src.transpose(1, 2)

        # 初始embedding
        src_embed = self.feature_embedding(src)  # [B, seq_len, hidden_size]

        # 位置編碼
        if self.batch_first:
            src_embed = self.pos_encoder(src_embed.transpose(0, 1))
            src_embed = src_embed.transpose(0, 1)
        else:
            src_embed = self.pos_encoder(src_embed)

        # embedding層的LN
        src_embed = self.embed_ln(src_embed)
        
        # 開始多次迭代 暫定為一次性 因為感覺過度的gate will get optimze result.
        for _ in range(self.num_iterations):            
            if self.batch_first:
                src_embed = self.transformer_encoder(src_embed) # [B, seq_len, hidden_size]
            else:
                src_embed = self.transformer_encoder(src_embed.transpose(0, 1)).transpose(0, 1)

            src_embed = self.gatemlp(src_embed)

        src_embed = src_embed.view(src_embed.size(0), -1)
        # 狀態值和優勢值
        value = self.fc_val(src_embed)       # [B, 1]
        advantage = self.fc_adv(src_embed)   # [B, num_actions]

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    


class mambaDuelingModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 nlayers: int,
                 num_actions: int,
                 time_features_in: int,
                 time_features_out: int = 32, # Time2Vec 的輸出維度，設為超參數
                 seq_dim: int = 300,
                 dropout: float = 0.1,
                 hidden_size: int = 96,
                 mode='full',
                 ssm_cfg: Optional[dict] = None,
                 moe_cfg: Optional[dict] = None,
                 ):

        super().__init__()
        self.time_embedding = SineActivation(in_features=time_features_in, out_features=time_features_out)
        self.dean = DAIN_Layer(mode=mode, input_dim=d_model) # DAIN 只處理市場數據

        self.market_embedding = nn.Linear(d_model, hidden_size)
        self.time_emb_projection = nn.Linear(time_features_out, hidden_size)
        
        # 門控層: 學習如何結合兩種資訊
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        # 最終的特徵轉換
        self.feature_embedding = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )


        # 狀態值網絡
        self.fc_val = nn.Sequential(
            nn.Linear(seq_dim * hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # 優勢網絡
        self.fc_adv = nn.Sequential(
            nn.Linear(seq_dim * hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions)
        )


        self.mixer = MixerModel(
            d_model= hidden_size,
            n_layer=nlayers,
            d_intermediate=256,
            dropout=dropout,
            ssm_cfg= ssm_cfg,
            moe_cfg=moe_cfg
        )

    def forward(self, src: Tensor, time_tau: Tensor) -> Tensor:
        time_emb = self.time_embedding(time_tau)
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
        src = self.feature_embedding(fused_emb)

        src, aux_loss = self.mixer(src)
        src = src.view(src.size(0), -1)

        value = self.fc_val(src)       # [B, 1]
        advantage = self.fc_adv(src)   # [B, num_actions]

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values  ,None, None

class mamba2DuelingModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 nlayers: int,
                 num_actions: int,
                 seq_dim: int = 300,
                 dropout: float = 0.1,
                 hidden_size :int = 64,
                 mode='full'):

        super().__init__()
        self.dean = DAIN_Layer(mode=mode, input_dim=d_model)

        # 狀態值網絡
        self.fc_val = nn.Sequential(
            nn.Linear(seq_dim * hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # 優勢網絡
        self.fc_adv = nn.Sequential(
            nn.Linear(seq_dim * hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions)
        )

        self.mixer = MixerModel(
            d_model= hidden_size,
            n_layer=nlayers,
            d_intermediate=1,
            ssm_cfg ={"layer": "Mamba2",},
            dropout=dropout
        )
        
        # 將資料映射至hidden_size維度
        self.feature_embedding = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, src: Tensor) -> Tensor:
        # src: [batch_size, seq_len, d_model]

        # 根據實測 rearrange 比較慢一些
        # src = rearrange(src,'b l d -> b d l')
                
        src = src.transpose(1, 2)        
        src = self.dean(src) # [B, seq_len, d_model]
        src = src.transpose(1, 2)
        src = self.feature_embedding(src)
        src = self.mixer(src)
        src = src.view(src.size(0), -1)

        # 狀態值和優勢值
        value = self.fc_val(src)       # [B, 1]
        advantage = self.fc_adv(src)   # [B, num_actions]

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # 門控殘差網絡 (GRN) - TFT 的基礎構建模塊
# class GatedResidualNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.context_dim = context_dim

#         # 如果有上下文向量，則加入
#         if self.context_dim is not None:
#             self.context_projection = nn.Linear(self.context_dim, self.hidden_dim)
        
#         self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
#         self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.elu = nn.ELU()
#         self.dropout = nn.Dropout(dropout)
        
#         # 門控機制
#         self.gate = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.Sigmoid()
#         )
        
#         # 殘差連接的線性轉換
#         if self.input_dim != self.output_dim:
#             self.residual_projection = nn.Linear(self.input_dim, self.output_dim)
#         else:
#             self.residual_projection = nn.Identity()
            
#         self.layer_norm = nn.LayerNorm(self.output_dim)

#     def forward(self, x, context=None):
#         residual = self.residual_projection(x)
        
#         x = self.fc1(x)
#         if context is not None:
#             context = self.context_projection(context)
#             x = x + context
            
#         x = self.elu(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
        
#         gate_val = self.gate(x)
#         x = x * gate_val
        
#         x = x + residual
#         x = self.layer_norm(x)
#         return x

# # 變數選擇網絡 (VSN)
# class VariableSelectionNetwork(nn.Module):
#     def __init__(self, input_dim, num_features, hidden_dim, dropout=0.1):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_features = num_features # 這就是您的 d_model
        
#         # 為每個特徵創建一個GRN
#         self.feature_grns = nn.ModuleList([
#             GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout) for _ in range(self.num_features)
#         ])
        
#         # 另一個GRN用於學習特徵權重
#         self.weight_grn = GatedResidualNetwork(
#             input_dim * hidden_dim, # 將所有特徵的GRN輸出展平
#             hidden_dim, 
#             self.num_features, 
#             dropout
#         )

#     def forward(self, x):
#         # x shape: [batch_size, seq_len, d_model]
        
#         # 1. 將每個特徵獨立輸入各自的GRN
#         #    需要先將特徵維度分離
#         split_features = torch.split(x, 1, dim=-1) # list of tensors, each [B, L, 1]
        
#         processed_features = []
#         for i in range(self.num_features):
#             processed_features.append(self.feature_grns[i](split_features[i]))
        
#         processed_features = torch.stack(processed_features, dim=-1) # [B, L, hidden_dim, num_features]
        
#         # 2. 學習特徵權重
#         flat_features = processed_features.view(x.size(0), x.size(1), -1) # [B, L, hidden_dim * num_features]
        
#         # 注意: TFT的VSN權重是在整個實體上共享的，這裡為了簡化，讓每個時間步都有自己的權重
#         feature_weights = self.weight_grn(flat_features) # [B, L, num_features]
#         feature_weights = F.softmax(feature_weights, dim=-1) # [B, L, num_features]
        
#         # 3. 加權特徵
#         # 原始特徵 x: [B, L, num_features]
#         # 權重 feature_weights: [B, L, num_features]
#         # 廣播機制會自動處理
#         weighted_features = x * feature_weights
        
#         return weighted_features, feature_weights

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def GLUFeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.Linear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

# class I2A_MambaDuelingModel(nn.Module):
#     def __init__(self,
#                  d_model: int,
#                  nlayers: int,
#                  num_actions: int,
#                  time_features_in: int,
#                  time_features_out: int = 32, # Time2Vec 的輸出維度，設為超參數
#                  seq_dim: int = 300,
#                  dropout: float = 0.1,
#                  hidden_size: int = 96,
#                  mode='full',
#                  ssm_cfg: Optional[dict] = None,
#                  moe_cfg: Optional[dict] = None,
#                  # --- 新增 I2A 相關參數 ---
#                  imagination_nlayers: int = 2,  # 想像路徑可以更淺
#                  num_imagined_features: int = 1 # 想像路徑要預測的特徵數量
#                  ):

#         super().__init__()
#         self.time_embedding = SineActivation(in_features=time_features_in, out_features=time_features_out)
#         self.dean = DAIN_Layer(mode=mode, input_dim=d_model) # DAIN 只處理市場數據
#         self.market_embedding = nn.Linear(d_model, hidden_size)        
#         self.time_emb_projection = nn.Linear(time_features_out, hidden_size)

#         self.mf_mixer = MixerModel( 
#             d_model= hidden_size*2,
#             n_layer=nlayers,
#             d_intermediate=256,
#             dropout=dropout,
#             ssm_cfg= ssm_cfg,
#             moe_cfg=moe_cfg
#         )

#         # === 路徑二：Model-Based (新增的想像路徑) ===
#         self.imagination_embedding = nn.Linear(d_model, hidden_size) # <--- 新增
        
#         self.imagination_mixer = MixerModel( # <--- 新增
#             d_model= hidden_size,
#             n_layer=imagination_nlayers, # 使用較淺的層數
#             d_intermediate=256,
#             dropout=dropout,
#             ssm_cfg= ssm_cfg, # 可以重用 ssm_cfg
#             moe_cfg=moe_cfg  # 可以重用 moe_cfg
#         )
        
#         # 想像路徑的預測頭 (MLP)：從序列最後一個時間點的輸出預測未來特徵
#         self.imagination_head = nn.Sequential( # <--- 新增
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 2, num_imagined_features),
#             nn.ReLU() # 0 - MAX
#         )

#         # === 融合決策層 (Dueling Heads) ===
#         # <--- 修改：輸入維度變為 (Model-Free 特徵 + 想像路徑特徵)
#         combined_input_dim = hidden_size *2 + num_imagined_features



#         # 狀態值網絡
#         self.fc_val = nn.Sequential(
#             nn.Linear(combined_input_dim, 512), # <--- 修改
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, 1)
#         )

#         # 優勢網絡
#         self.fc_adv = nn.Sequential(
#             nn.Linear(combined_input_dim, 512), # <--- 修改
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_actions)
#         )

#         self.ffn = GLUFeedForward(
#             dim = hidden_size*2,
#             mult = 4,
#             dropout = dropout
#         )

#     def forward(self, src: Tensor, time_tau: Tensor) -> tuple[Tensor, Tensor, Tensor]:
#         # --- 數據預處理 (共享) ---
#         # 歸一化市場數據 [B, L, D] -> [B, D, L] -> [B, D, L] -> [B, L, D]

#         market_data_normalized = src.transpose(1, 2)
#         market_data_normalized = self.dean(market_data_normalized)
#         market_data_normalized = market_data_normalized.transpose(1, 2)
#         market_emb = self.market_embedding(market_data_normalized) # [B, L, hidden_size]


#         # 時間資訊處理
#         time_emb = self.time_embedding(time_tau)
#         time_emb_proj = self.time_emb_projection(time_emb) # [B, L, hidden_size]
        

#         # 融合特徵        
#         processed_emb = self.ffn(torch.cat([market_emb, time_emb_proj], dim=-1))


#         # Model-Free Mixer
#         mf_seq_out, mf_aux_loss = self.mf_mixer(processed_emb)
#         mf_features = mf_seq_out[:, -1, :]


#         # --- 路徑二：Model-Based (想像) ---
#         imag_emb = self.imagination_embedding(market_data_normalized)

#         imag_seq_out, imag_aux_loss = self.imagination_mixer(imag_emb) # [B, L, hidden_size]
        

#         # 只取序列的最後一個時間點的輸出來預測未來
#         imag_last_step = imag_seq_out[:, -1, :] # [B, hidden_size]
#         imagined_features = self.imagination_head(imag_last_step) # [B, num_imagined_features]
        

#         # --- 融合 (Fusion) ---
#         combined_features = torch.cat([mf_features, imagined_features], dim=1) # [B, (L*hidden) + num_imagined]

#         # --- Dueling 決策 ---
#         value = self.fc_val(combined_features)      # [B, 1]
#         advantage = self.fc_adv(combined_features)  # [B, num_actions]

#         q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        

        
#         # 返回 Q-values, Mamba 輔助損失, 以及 "想像的" 預測值 (用於計算想像損失)
#         return q_values, None, imagined_features



class I2A_MambaDuelingModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 nlayers: int,
                 num_actions: int,
                 time_features_in: int,
                 time_features_out: int = 32, # Time2Vec 的輸出維度，設為超參數
                 seq_dim: int = 300,
                 dropout: float = 0.1,
                 hidden_size: int = 96,
                 mode='full',
                 ssm_cfg: Optional[dict] = None,
                 moe_cfg: Optional[dict] = None,
                 # --- 新增 I2A 相關參數 ---
                 imagination_nlayers: int = 2,  # 想像路徑可以更淺
                 num_imagined_features: int = 1 # 想像路徑要預測的特徵數量
                 ):

        super().__init__()
        self.time_embedding = SineActivation(in_features=time_features_in, out_features=time_features_out)
        self.dean = DAIN_Layer(mode=mode, input_dim=d_model) # DAIN 只處理市場數據
        self.market_embedding = nn.Linear(d_model, hidden_size)        
        self.time_emb_projection = nn.Linear(time_features_out, hidden_size)

        self.mf_mixer = MixerModel( 
            d_model= hidden_size*2,
            n_layer=nlayers,
            d_intermediate=256,
            dropout=dropout,
            ssm_cfg= ssm_cfg,
            moe_cfg=moe_cfg
        )

        # === 融合決策層 (Dueling Heads) ===
        # <--- 修改：輸入維度變為 (Model-Free 特徵)
        combined_input_dim = hidden_size *2 



        # 狀態值網絡
        self.fc_val = nn.Sequential(
            nn.Linear(combined_input_dim, 512), # <--- 修改
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # 優勢網絡
        self.fc_adv = nn.Sequential(
            nn.Linear(combined_input_dim, 512), # <--- 修改
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions)
        )

        self.ffn = GLUFeedForward(
            dim = hidden_size*2,
            mult = 4,
            dropout = dropout
        )

    def forward(self, src: Tensor, time_tau: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # --- 數據預處理 (共享) ---
        # 歸一化市場數據 [B, L, D] -> [B, D, L] -> [B, D, L] -> [B, L, D]
        market_data_normalized = src.transpose(1, 2)
        market_data_normalized = self.dean(market_data_normalized)
        market_data_normalized = market_data_normalized.transpose(1, 2)
        market_emb = self.market_embedding(market_data_normalized) # [B, L, hidden_size]


        # 時間資訊處理
        time_emb = self.time_embedding(time_tau)
        time_emb_proj = self.time_emb_projection(time_emb) # [B, L, hidden_size]
        

        # 融合特徵        
        processed_emb = self.ffn(torch.cat([market_emb, time_emb_proj], dim=-1))


        # Model-Free Mixer
        mf_seq_out, mf_aux_loss = self.mf_mixer(processed_emb)
        mf_features = mf_seq_out[:, -1, :]



        # --- Dueling 決策 ---
        value = self.fc_val(mf_features)      # [B, 1]
        advantage = self.fc_adv(mf_features)  # [B, num_actions]

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))       

        
        
        # 返回 Q-values, Mamba 輔助損失, 以及 "想像的" 預測值 (用於計算想像損失)
        return q_values, None, None
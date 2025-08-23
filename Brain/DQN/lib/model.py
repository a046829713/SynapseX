
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder
from Brain.Common.transformer_tool import TransformerEncoderLayer, PositionalEncoding
from Brain.Common.dain import DAIN_Layer
import time
from einops import rearrange
from Brain.Common.ssm_tool import MixerModel,GatedMLP
from torch.nn import functional as F
from typing import Optional

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
                 seq_dim: int = 300,
                 dropout: float = 0.1,
                 mode='full',
                 ssm_cfg: Optional[dict] = None,
                 moe_cfg: Optional[dict] = None
                 ):

        super().__init__()
        self.dean = DAIN_Layer(mode=mode, input_dim=d_model)

        # 狀態值網絡
        self.fc_val = nn.Sequential(
            nn.Linear(seq_dim * d_model, 512),
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
            nn.Linear(seq_dim * d_model, 512),
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
            d_model= d_model,
            n_layer=nlayers,
            d_intermediate=256,
            dropout=dropout,
            ssm_cfg= ssm_cfg,
            moe_cfg=moe_cfg
        )
        

    def forward(self, src: Tensor) -> Tensor:
        # src: [batch_size, seq_len, d_model]

        # 根據實測 rearrange 比較慢一些
        # src = rearrange(src,'b l d -> b d l')        
        src = src.transpose(1, 2)        
        src = self.dean(src) # [B, seq_len, d_model]
        src = src.transpose(1, 2)
        

        src, aux_loss = self.mixer(src)



        src = src.view(src.size(0), -1)

        # 狀態值和優勢值
        value = self.fc_val(src)       # [B, 1]
        advantage = self.fc_adv(src)   # [B, num_actions]

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values, aux_loss



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
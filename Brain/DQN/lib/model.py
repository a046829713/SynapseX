
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder
from Brain.Common.transformer_tool import TransformerEncoderLayer, PositionalEncoding
from Brain.Common.dain import DAIN_Layer
import time


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
                 mode='full'):
        """
            原本EncoderLayer 是使用官方的，後面因為訓練上難以收斂
            故重新在製作一次屬於自己的TransformerEncoderLayer
        """

        super().__init__()
        self.batch_first = batch_first
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layers = TransformerEncoderLayer(
            hidden_size, nhead, d_hid, dropout, batch_first=self.batch_first)

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers, norm=nn.LayerNorm(hidden_size), enable_nested_tensor=False)

        self.dean = DAIN_Layer(mode=mode,input_dim=d_model)

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

        # 將資料映射
        self.embedding = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.embed_ln = nn.LayerNorm(hidden_size)  # 層歸一化

        # 初始化權重
        # self.apply(self._init_weights)
        

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He初始化
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, d_model]``

        Returns:
            output Tensor of shape ``[batch_size, num_actions]``

        """
        # 
        src = src.transpose(1, 2)
        src = self.dean(src)
        src = src.transpose(1, 2)

        src = self.embedding(src)

        # src = torch.Size([1, 300, 6])
        if self.batch_first:
            src = self.pos_encoder(src.transpose(0, 1))
        else:
            src = self.pos_encoder(src)

        src = self.embed_ln(src.transpose(0, 1))

        if self.batch_first:
            output = self.transformer_encoder(src)
        else:
            output = self.transformer_encoder(src.transpose(0, 1))

        # output = torch.Size([1, 300, 6])
        x = self.linear(output)
        x = x.view(x.size(0), -1)

        # 狀態值和優勢值
        value = self.fc_val(x)
        advantage = self.fc_adv(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        # 計算最終的Q值

        return q_values
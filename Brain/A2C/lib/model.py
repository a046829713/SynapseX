import torch
from torch import nn
from Brain.Common.transformer_tool import TransformerEncoderLayer, PositionalEncoding
from torch.nn import TransformerEncoder
from torch import nn, Tensor
import time
from Brain.Common.dain import DAIN_Layer


class TransformerModel(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 n_actions: int,
                 hidden_size: int,
                 dropout: float = 0.1,
                 batch_first=True,
                 mode='full'
                 ) -> None:
        """
            原本EncoderLayer 是使用官方的，後面因為訓練上難以收斂
            故重新在製作一次屬於自己的TransformerEncoderLayer,並且加入chain of thought

            d_model = 6          # 输入特征的维度
            nhead = 4            # 多头注意力机制的头数
            d_hid = 128          # 前馈网络的隐藏层大小
            nlayers = 2          # Transformer Encoder 的层数
            n_actions = 3       # 动作的数量
            hidden_size = 64     # 嵌入后的特征维度
            seq_dim = 300        # 序列长度
            dropout = 0.1        # Dropout 概率
            batch_first = True   # 是否将 batch 维度放在第一个维度
            num_iterations = 3   # 迭代次数（思维链的长度）

            為甚麼輸出的時候不使用softmax 是因為CrossEntropy 裡面已經會使用到了。
            一開始在測試chain of thought 發現因為TransformerEncoderLayer 裡面一開始採用
            alpha = 0, 所以輸入和輸出會都一樣。
        """
        super(TransformerModel, self).__init__()

        # 將資料映射
        self.embedding = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.batch_first = batch_first
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layers = TransformerEncoderLayer(
            hidden_size, nhead, d_hid, dropout, batch_first=self.batch_first)

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers, norm=nn.LayerNorm(hidden_size), enable_nested_tensor=False)

        self.embed_ln = nn.LayerNorm(hidden_size)  # 層歸一化

        # 產生各動作選擇的機率
        self.policy_head = nn.Sequential(
            nn.Linear(2400, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

        # Critic 的 Value Head
        self.value_head = nn.Sequential(
            nn.Linear(2400, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.dean = DAIN_Layer(mode=mode, input_dim=d_model)


        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, hidden_size // 8)
        )
        
    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, d_model]``

        Returns:
            output Tensor of shape ``[batch_size, num_actions]``

        """
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
            src = self.transformer_encoder(src)
        else:
            src = self.transformer_encoder(src.transpose(0, 1))

        src = self.linear(src)
        src = src.view(src.size(0), -1)

        policy_logits = self.policy_head(src)


        # Critic 输出
        state_value = self.value_head(src.detach())

        return policy_logits, state_value


class ActorCriticModel(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 n_actions: int,
                 hidden_size: int,
                 dropout: float = 0.1,
                 batch_first=True,
                 num_iterations=3,
                 mode='full'):
        """

        """
        super(ActorCriticModel, self).__init__()
        self.dean = DAIN_Layer(mode=mode, input_dim=d_model)

        self.n_actions = n_actions

        # 決定迭代的次數
        self.num_iterations = num_iterations

        self.transformer = TransformerModel(d_model=d_model,
                                            nhead=nhead,
                                            d_hid=d_hid,
                                            nlayers=nlayers,
                                            n_actions=n_actions,
                                            hidden_size=hidden_size,
                                            dropout=dropout,
                                            batch_first=batch_first)
        self.batch_first = batch_first

    def forward(self, src: Tensor) -> Tensor:
        """
            Arguments:
                src: Tensor, shape ``[batch_size, seq_len, d_model]`` if batch_first is True,
                    else ``[seq_len, batch_size, d_model]``

            Returns:
                policy_logits: Tensor of shape ``[batch_size, n_actions]``
                state_value: Tensor of shape ``[batch_size, 1]``        
        """
        # The sequence length needs to be calculated together, so the second dimension should be swapped with the first dimension.
        src = src.transpose(1, 2)
        src = self.dean(src)
        src = src.transpose(1, 2)

        # torch.Size([2, 300, 16])
        if self.batch_first:
            batch_size, seq_len, _ = src.size()
        else:
            seq_len, batch_size, _ = src.size()

        # 初始化 new_input，添加全零的动作和状态值信息
        zeros_actions = torch.zeros(
            batch_size, seq_len, self.n_actions, device=src.device)
        zeros_values = torch.zeros(batch_size, seq_len, 1, device=src.device)
        new_input = torch.cat((src, zeros_actions, zeros_values), dim=2)

        
        for _ in range(self.num_iterations):
            policy_logits, state_value = self.transformer(new_input)
            # 将 policy_logits 和 state_value 扩展到序列长度
            policy_logits_expanded = policy_logits.unsqueeze(
                1).expand(-1, seq_len, -1)

            state_value_expanded = state_value.unsqueeze(
                1).expand(-1, seq_len, -1)
            # 更新 new_input
            new_input = torch.cat(
                (src, policy_logits_expanded, state_value_expanded), dim=2)

        return policy_logits, state_value

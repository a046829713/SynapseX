import torch
from torch import nn
from Common.transformer_tool import TransformerEncoderLayer, PositionalEncoding
from torch.nn import TransformerEncoder
from torch import nn, Tensor


class ActorCriticModel(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 n_actions: int,
                 hidden_size: int,
                 seq_dim: int = 300,
                 dropout: float = 0.5,
                 batch_first=True,
                 num_iterations=3):
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
        super(ActorCriticModel, self).__init__()

        # 決定迭代的次數
        self.num_iterations = num_iterations

        self.batch_first = batch_first
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layers = TransformerEncoderLayer(
            hidden_size, nhead, d_hid, dropout, batch_first=self.batch_first)

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers, norm=nn.LayerNorm(hidden_size), enable_nested_tensor=False)

        self.policy_head = nn.Linear(hidden_size, n_actions)

        # Critic 的 Value Head
        self.value_head = nn.Linear(hidden_size, 1)

        # 將資料映射
        self.embedding = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.embed_ln = nn.LayerNorm(hidden_size)  # 層歸一化

        # 初始化權重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, d_model]``

        Returns:
            output Tensor of shape ``[batch_size, num_actions]``

        """
        src = self.embedding(src)

        # src = torch.Size([1, 300, 6])
        if self.batch_first:
            src = self.pos_encoder(src.transpose(0, 1))
        else:
            src = self.pos_encoder(src)

        src = self.embed_ln(src.transpose(0, 1))

        # torch.Size([32, 300, 64])
        for _ in range(self.num_iterations):
            if self.batch_first:
                src = self.transformer_encoder(src)
            else:
                src = self.transformer_encoder(src.transpose(0, 1))

        x = src.mean(dim=1)

        policy_logits = self.policy_head(x)

        # Critic 输出
        state_value = self.value_head(x)

        return policy_logits, state_value
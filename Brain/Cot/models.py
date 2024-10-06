import torch
import torch.nn as nn
from Common.transformer_tool import TransformerEncoderLayer, PositionalEncoding

# Self-Attention Transformer Encoder Layer
class TransformerPolicy(nn.Module):
    def __init__(self, 
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 num_actions: int,
                 hidden_size: int,
                 seq_dim: int = 300,
                 dropout: float = 0.5,
                 batch_first=True):
        super(TransformerPolicy, self).__init__()
        
        self.batch_first = batch_first        
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        self.embedding = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        encoder_layers = TransformerEncoderLayer(
            hidden_size, nhead, d_hid, dropout, batch_first=self.batch_first)

        self.transformer = nn.TransformerEncoder(
            encoder_layers,
            num_layers=nlayers,
        )
        self.fc_out = nn.Linear(hidden_size, num_actions)  # 修改為 hidden_size
        self.softmax = nn.Softmax(dim=-1)  # 將得分轉換為概率

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        action_probs = self.softmax(x)
        return action_probs

# Chain of Thought 模型，用於逐步推理策略
class ChainOfThoughtPolicy(nn.Module):
    def __init__(self, transformer_policy, thought_steps=3):
        super(ChainOfThoughtPolicy, self).__init__()
        self.transformer_policy = transformer_policy
        self.thought_steps = thought_steps

    def forward(self, x):
        thought_process = []
        for step in range(self.thought_steps):
            action_probs = self.transformer_policy(x)
            thought_process.append(action_probs)
            # 更新輸入 x，將 action_probs 納入下一步的輸入
            x = self.update_input_with_thought(x, action_probs)
        # 將多個思維步驟的結果進行聚合（例如取平均）
        final_action_probs = sum(thought_process) / self.thought_steps
        return final_action_probs

    def update_input_with_thought(self, x, action_probs):
        # 假設 x 的形狀為 [batch_size, seq_len, feature_dim]
        # action_probs 的形狀為 [batch_size, seq_len, num_actions]
        # 我們將 action_probs 作為新的特徵，與原始輸入 x 進行拼接
        updated_x = torch.cat((x, action_probs), dim=-1)  # 在特徵維度上拼接
        return updated_x

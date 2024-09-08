import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from SynapseX.Brain.Common.transformer_tool import TransformerEncoderLayer, PositionalEncoding
from torch.nn import TransformerEncoder
# from torchvision.models import efficientnet_v2_s

class Actor(nn.Module):
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
        """
            原本EncoderLayer 是使用官方的，後面因為訓練上難以收斂
            故重新在製作一次屬於自己的TransformerEncoderLayer
            
            移除 Q 值计算：

                A2C 中的 Actor 不负责计算 Q 值，因此移除了原来的 fc_val 和 fc_adv 网络部分。
        """

        super().__init__()
        self.batch_first = batch_first
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layers = TransformerEncoderLayer(
            hidden_size, nhead, d_hid, dropout, batch_first=self.batch_first)

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers, norm=nn.LayerNorm(hidden_size), enable_nested_tensor=False)

        # 策略网络
        self.fc_policy = nn.Sequential(
            nn.Linear(seq_dim * hidden_size // 8, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions),
            
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

        # 将数据映射
        self.embedding = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.embed_ln = nn.LayerNorm(hidden_size)  # 层归一化

        # 初始化权重
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
            output Tensor of shape ``[batch_size, num_actions]``, 动作的概率分布
        """
        src = self.embedding(src)

        if self.batch_first:
            src = self.pos_encoder(src.transpose(0, 1))
        else:
            src = self.pos_encoder(src)

        src = self.embed_ln(src.transpose(0, 1))

        if self.batch_first:
            output = self.transformer_encoder(src)
        else:
            output = self.transformer_encoder(src.transpose(0, 1))

        x = self.linear(output)
        x = x.view(x.size(0), -1)

        # 输出策略分布
        policy = torch.sigmoid(self.fc_policy(x))

        return policy
    
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
    
# class Actor(nn.Module):

#     def __init__(self,
#                  in_channels:int,
#                  action_size:float,
#                  dropout:float = 0.2):
        
#         super().__init__()
        
#         # 加载预训练的EfficientNet模型
#         self.model = efficientnet_v2_s(weights=None)
        
#         # 修改第一层卷积的输入通道数
#         # EfficientNet的第一层卷积通常命名为 'features.0.0'
#         first_conv = self.model.features[0][0]
#         new_first_conv = nn.Conv2d(
#             in_channels=in_channels, # 為了製作通用模型
#             out_channels=first_conv.out_channels,
#             kernel_size=first_conv.kernel_size,
#             stride=first_conv.stride,
#             padding=first_conv.padding,
#             bias=False
#         )

#         # 替换模型的第一层卷积
#         self.model.features[0][0] = new_first_conv
        
#         # 輸出動作網絡
#         self.action_layer = nn.Sequential(
#             nn.Linear(1000, 1024),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(1024, action_size),
            
#         )
        
#     def forward(self, src: Tensor) -> Tensor:
#         x = self.model(src)
#         x = torch.sigmoid(self.action_layer(x)) # size() = 1,1
#         return x
    






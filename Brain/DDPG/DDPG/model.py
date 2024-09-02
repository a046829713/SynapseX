import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s
import torch.nn as nn
from torch import Tensor

class Actor(nn.Module):
    def __init__(self,
                 in_channels:int,
                 action_size:float,
                 dropout:float = 0.2):
        
        super().__init__()
        
        # 加载预训练的EfficientNet模型
        self.model = efficientnet_v2_s(weights=None)
        
        # 修改第一层卷积的输入通道数
        # EfficientNet的第一层卷积通常命名为 'features.0.0'
        first_conv = self.model.features[0][0]
        new_first_conv = nn.Conv2d(
            in_channels=in_channels, # 為了製作通用模型
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )

        # 替换模型的第一层卷积
        self.model.features[0][0] = new_first_conv
        
        # 輸出動作網絡
        self.action_layer = nn.Sequential(
            nn.Linear(1000, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, action_size),
            
        )
        
    def forward(self, src: Tensor) -> Tensor:
        x = self.model(src)
        x = torch.sigmoid(self.action_layer(x)) # size() = 1,1
        return x
    


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



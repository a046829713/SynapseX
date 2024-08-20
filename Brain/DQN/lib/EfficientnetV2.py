import torch
from torchvision.models import efficientnet_v2_s
import torch.nn as nn
from torch import Tensor
import time

class EfficientnetV2SmallDuelingModel(nn.Module):
    def __init__(self,
                 in_channels:int,
                 num_actions:int,
                 dropout = 0.2):
        """

        """
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


        # 狀態值網絡
        self.fc_val = nn.Sequential(
            nn.Linear( 1000, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1)
        )

        # 優勢網絡
        self.fc_adv = nn.Sequential(
            nn.Linear(1000, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_actions)
        )

    def forward(self, src: Tensor) -> Tensor:
        """

        """
        x = self.model(src)
        
        value = self.fc_val(x)
        # 狀態值和優勢值
        advantage = self.fc_adv(x)

        # 計算最終的Q值
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# 测试新的输入大小
# x = torch.randn(16, 1, 6, 300)
# model = EfficientnetV2SmallDuelingModel(in_channels=1,num_actions=3)
# output = model(x)
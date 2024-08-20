import torch
import torch.nn as nn
from torchsummary import summary
import time

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer=nn.SiLU, gate_layer=nn.Sigmoid):
        super(SqueezeExcite, self).__init__()
        self.gate_layer = gate_layer()
        self.rd_channels = rd_channels or max(1, int(in_chs * rd_ratio))
        
        self.conv_reduce = nn.Conv2d(in_chs, self.rd_channels, 1, bias=True)
        self.act1 = act_layer()
        
        self.conv_expand = nn.Conv2d(self.rd_channels, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_layer(x_se)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride, act_layer=nn.SiLU):
        super(DepthwiseSeparableConv, self).__init__()
        self.conv_dw = nn.Conv2d(in_chs, in_chs, kernel_size, stride=stride, padding=kernel_size // 2, groups=in_chs, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chs)
        self.act1 = act_layer()

        self.se = SqueezeExcite(in_chs)

        self.conv_pw = nn.Conv2d(in_chs, out_chs, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs)
        self.act2 = act_layer()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        return self.act2(x)

class InvertedResidual(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride, exp_ratio=6, se_ratio=0.25, drop_path_rate=0.0, act_layer=nn.SiLU):
        super(InvertedResidual, self).__init__()
        self.has_se = se_ratio is not None and se_ratio > 0
        self.drop_path_rate = drop_path_rate
        self.exp_channels = in_chs * exp_ratio
        self.conv_pw = nn.Conv2d(in_chs, self.exp_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.exp_channels)
        self.act1 = act_layer()
        self.conv_dw = nn.Conv2d(self.exp_channels, self.exp_channels, kernel_size, stride=stride, padding=kernel_size // 2, groups=self.exp_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(self.exp_channels)
        self.act2 = act_layer()
        self.se = SqueezeExcite(self.exp_channels, se_ratio) if self.has_se else nn.Identity()
        self.conv_pwl = nn.Conv2d(self.exp_channels, out_chs, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chs)
        self.act3 = act_layer()

    def forward(self, x):
        residual = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.drop_path_rate > 0.0:
            x = self.drop_path(x)
        if residual.size() == x.size():
            x += residual
        return x

class EfficientNetB3(nn.Module):
    def __init__(self, actions_n):
        super(EfficientNetB3, self).__init__()

        # Stem
        self.conv_stem = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=3, stride=2, padding=1, bias=False)
        
        # 透過通道 歸一化後進行縮放和平移
        self.bn1 = nn.BatchNorm2d(40)

        # 在深層的模型效果較佳
        self.act1 = nn.SiLU(inplace=True)

        # Blocks
        self.blocks = nn.Sequential(
            # Stage 1
            DepthwiseSeparableConv(40, 24, 3, 1),
            DepthwiseSeparableConv(24, 24, 3, 1),
            # Stage 2
            InvertedResidual(24, 32, 3, 2, exp_ratio=6),
            InvertedResidual(32, 32, 3, 1, exp_ratio=6),
            InvertedResidual(32, 32, 3, 1, exp_ratio=6),
            # Stage 3
            InvertedResidual(32, 48, 5, 2, exp_ratio=6),
            InvertedResidual(48, 48, 5, 1, exp_ratio=6),
            InvertedResidual(48, 48, 5, 1, exp_ratio=6),
            InvertedResidual(48, 48, 5, 1, exp_ratio=6),
            # Stage 4
            InvertedResidual(48, 96, 3, 2, exp_ratio=6),
            InvertedResidual(96, 96, 3, 1, exp_ratio=6),
            InvertedResidual(96, 96, 3, 1, exp_ratio=6),
            InvertedResidual(96, 96, 3, 1, exp_ratio=6),
            # Stage 5
            InvertedResidual(96, 136, 5, 1, exp_ratio=6),
            InvertedResidual(136, 136, 5, 1, exp_ratio=6),
            InvertedResidual(136, 136, 5, 1, exp_ratio=6),
            InvertedResidual(136, 136, 5, 1, exp_ratio=6),
            # Stage 6
            InvertedResidual(136, 232, 5, 2, exp_ratio=6),
            InvertedResidual(232, 232, 5, 1, exp_ratio=6),
            InvertedResidual(232, 232, 5, 1, exp_ratio=6),
            InvertedResidual(232, 232, 5, 1, exp_ratio=6),
            # Stage 7
            InvertedResidual(232, 384, 3, 1, exp_ratio=6),
            InvertedResidual(384, 384, 3, 1, exp_ratio=6),
            InvertedResidual(384, 384, 3, 1, exp_ratio=6),
            InvertedResidual(384, 384, 3, 1, exp_ratio=6),
        )

        # Head
        self.conv_head = nn.Conv2d(384, 1536, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1536)
        self.act2 = nn.SiLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        
        
        self.fc_val = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Linear(1024, actions_n)
        )

    def forward(self, x):
        # torch.Size([1, 1, 6, 300])
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

# 創建自訂義 EfficientNet-B3 模型
# input_shape = (1, 6, 300)  # 假設新的輸入張量尺寸
# num_classes = 3
# model = EfficientNetB3(num_classes)

# 打印修改後的模型結構
# print(model)

# 創建一個隨機輸入張量，假設圖像尺寸為 1x6x300
# input_tensor = torch.randn(1, 1, 6, 300) # (batch_size, channel, height, width)

# 前向傳播
# output = model(input_tensor)

# print(output)
# 打印模型摘要
# summary(model, input_shape)

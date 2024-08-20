from torch import nn
import torch
import numpy as np
import time

class GradientPolicy(nn.Module):
    def __init__(self):
        """DDPG policy network initializer."""
        super().__init__()
        # [batch_size, features, height, width]
        # 對於圖像來說，這四個維度分別代表了批次大小、
        # 顏色通道數、圖像高度和圖像寬度。
        # 例如，一批含有32個彩色圖像（每個圖像有3個顏色通道，圖像大小為64x64像素）的輸入將會有形狀 [32, 3, 64, 64]。

        # [batch_size, channels, height, width]
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1, 48)),
            nn.ReLU()
        )

        self.final_convolution = nn.Conv2d(
            in_channels=21, out_channels=1, kernel_size=(1, 1))

        
        self.softmax = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
    def mu(self, observation, last_action):
        """Defines a most favorable action of this policy given input x.

        Args:
          observation: environment observation .
          last_action: Last action performed by agent.

        Returns:
          Most favorable action.
        """
        if isinstance(observation, np.ndarray):
            # torch.Size([1, 3, 4, 50]) batch_size,featrue_number,asseets,window
            observation = torch.from_numpy(observation).to(self.device)             
            
        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action).to(self.device)

        last_stocks, cash_bias = self._process_last_action(last_action)        
        
        # shape [N, PORTFOLIO_SIZE + 1, 19, 1]
        output = self.sequential(observation)        

        # shape [N, 21, PORTFOLIO_SIZE, 1]
        output = torch.cat([last_stocks, output ], dim=1)
        
        # shape [N, 1, PORTFOLIO_SIZE, 1]
        output = self.final_convolution(output)

        # shape [N, 1, PORTFOLIO_SIZE + 1, 1]
        # original author
        # output = torch.cat([output, cash_bias], dim=2)
        # 根據他別的庫 證明cash_bias 我是對的
        output = torch.cat([cash_bias ,output ], dim=2)


        # output shape must be [N, features] = [1, PORTFOLIO_SIZE + 1], being N batch size (1)
        # and size the number of features (weights vector).
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # shape [N, PORTFOLIO_SIZE + 1]

        
        output = self.softmax(output)        
        return output

    def forward(self, observation, last_action):
        """Policy network's forward propagation.

        Args:
          observation: environment observation (dictionary).
          epsilon: exploration noise to be applied.

        Returns:
          Action to be taken (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1        
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias

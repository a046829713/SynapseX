import numpy as np


a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])


print(np.dot(a,b))

import torch
from torch import nn

# 定義兩個向量
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 計算點積
dot_product = torch.dot(a, b)
print(dot_product)  # 輸出：32.0

linear_layer = nn.Linear(input_dim, embed_dim)
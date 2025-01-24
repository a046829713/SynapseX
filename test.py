import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

# 1. 創建數據集
# 生成一些隨機的線性數據
torch.manual_seed(0)
x = torch.rand(100, 1) * 10  # 特徵範圍在 [0, 10)


y = 5 * x + 3 + torch.randn(100, 1)  # 目標值 y = 5x + 3 加上少量隨機噪聲

# plt.plot(x)
# plt.plot(y)
# plt.show()

# 2. 定義模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1 )



    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 3. 定義損失函數和優化器
criterion = nn.MSELoss()



# 使用 L2 正則化 (weight_decay 參數實現)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # weight_decay 即為 L2 正則化參數

# 4. 訓練模型
epochs = 1000
for epoch in range(epochs):
    # 前向傳播
    predictions = model(x)
    loss = criterion(predictions, y)

    # 反向傳播與優化
    optimizer.zero_grad()
    loss.backward()

    # 目前是第幾次:99 梯度為: tensor([-0.9966]) 權重為: Parameter containing:
    # tensor([1.5611], requires_grad=True)
    for i in model.parameters():
        if epoch<100:
            print(f"目前是第幾次:{epoch}","梯度為:",i.grad,"權重為:",i)


    optimizer.step()
    if epoch==100:
        time.sleep(100)
        
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# # 5. 測試模型
# with torch.no_grad():
#     test_x = torch.tensor([[4.0], [7.0], [9.0]])
#     test_y = model(test_x)
#     print("\nTest Results:")
#     for i, value in enumerate(test_x):
#         print(f"Input: {value.item()}, Predicted Output: {test_y[i].item():.4f}")

# # 6. 檢查模型的參數
# print("\nModel Parameters:")
# for name, param in model.named_parameters():
#     print(f"{name}: {param.data}")

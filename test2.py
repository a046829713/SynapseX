import torch


class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        print(result)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        print("目前ctx 裡面為？")
        print(result)
        print('*'*120)
        return grad_output * result
    
# 建立一個輸入 tensor，並設定 requires_grad=True 以便追蹤其梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 使用自訂的 Exp 函數進行計算
y = Exp.apply(x)

# 假設我們計算一個簡單的 loss，例如所有輸出值的總和
loss = y.sum()

# 反向傳播
loss.backward()

# print("輸入 x:", x)
# print("指數運算結果 y:", y)
# print("x 的梯度:", x.grad)


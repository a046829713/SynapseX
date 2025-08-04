import torch


class ExpAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        output = i.exp() + i

        ctx.save_for_backward(i)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """        
            在反向傳播中，我們根據鏈式法則計算梯度。
            梯度 = 上游梯度 * 本地梯度
            本地梯度是 d/di(exp(i) + i) = exp(i) + 1

        Args:
            ctx (_type_): _description_
            grad_output (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 1. 取出儲存的張量
        i, = ctx.saved_tensors

        # 2. 計算本地梯度 (local gradient)
        local_grad = i.exp() + 1

        # 3. 根據鏈式法則，將上游梯度 (grad_output) 乘上本地梯度
        grad_input = grad_output * local_grad

        return grad_input
    
# 建立一個輸入 tensor，並設定 requires_grad=True 以便追蹤其梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 使用自訂的 Exp 函數進行計算
y = ExpAdd.apply(x)


# 假設我們計算一個簡單的 loss，例如所有輸出值的總和
loss = y.sum()

print("loss:", loss)

# 反向傳播
loss.backward()

print(x.grad)

# print("輸入 x:", x)
# print("指數運算結果 y:", y)
# print("x 的梯度:", x.grad)


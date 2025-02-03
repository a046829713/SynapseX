import torch

# 繼承自 Function
class LinearFunction(torch.autograd.Function):

    # 注意：forward、setup_context 以及 backward 都是 @staticmethods
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs 是一個 Tuple，包含所有傳入 forward 的輸入。
    # output 是 forward() 的輸出。
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    # 此函數只有一個輸出，因此只接收一個梯度
    @staticmethod
    def backward(ctx, grad_output):
        # 這是一種非常方便的模式 —— 在 backward 開頭處解包 saved_tensors 並將所有對輸入的梯度初始化為 None。
        # 多餘的 None 返回值會被忽略，因此即使函數有可選的輸入，return 語句也可以保持簡單。
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # 這些 needs_input_grad 檢查是可選的，主要用來提高效率。如果你想讓代碼更簡單，可以跳過它們。
        # 為不需要梯度的輸入返回梯度並不會造成錯誤。
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

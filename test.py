import torch
import torch.nn as nn
from torch import Tensor
import torch._dynamo # 引入 dynamo
from Brain.DQN.lib import environment, common, model
import time

def test_grad(eager_model, compiled_model, dummy_input):
    print(id(eager_model))
    print(id(compiled_model))
    eager_model.eval()
    output_eager = eager_model(dummy_input)
    loss_eager = output_eager.sum() # 使用一個簡單的 loss
    loss_eager.backward()
    # 儲存 eager 模式的梯度
    grads_eager = {name: p.grad.clone() for name, p in eager_model.named_parameters()}


    eager_model.zero_grad()


    # --- 在 Compiled 模式下計算梯度 ---
    compiled_model.eval()
    output_compiled = compiled_model(dummy_input)
    loss_compiled = output_compiled.sum()
    loss_compiled.backward()

    # 儲存 compiled 模式的梯度
    grads_compiled = {name: p.grad.clone() for name, p in compiled_model._orig_mod.named_parameters()}
    compiled_model.zero_grad()


    # --- 比較梯度 ---
    all_correct = True
    for name in grads_eager.keys():
        if not torch.allclose(grads_eager[name], grads_compiled[name], atol=1e-6):
            print(f"梯度不匹配於: {name}")
            print(f"Eager grad: {grads_eager[name]}")
            print(f"Compiled grad: {grads_compiled[name]}")
            all_correct = False

    if all_correct:
        print("所有梯度驗證成功！Compiled 和 Eager 模式下的梯度一致。")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


moe_config = {"num_experts": 16}
ssm_cfg = {"expand": 4}

net = model.mambaDuelingModel(
    d_model=12,
    nlayers=6,
    num_actions=3,
    seq_dim=300,
    dropout=0.3,
    ssm_cfg=ssm_cfg,
    moe_cfg=moe_config,
).to(DEVICE)




test_grad(net, torch.compile(net), torch.randn(32, 300, 12).to(DEVICE))





# net.compile()

# # 2. 建立一個假的輸入 Tensor，維度要和實際輸入一樣
# dummy_input = torch.randn(32, 300, 12).to(DEVICE) # (batch_size, seq_len, d_model)

# import time

# for i in  range(100):
#     start = time.time()
#     net(dummy_input)


#     print("本次使用時間：",time.time() - start)


# 3. 使用 explain 執行！
# 這會執行一次 forward pass 並印出詳細的編譯報告
# explanation_output = torch._dynamo.explain(net)(dummy_input)
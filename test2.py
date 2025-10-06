import torch
import torch.nn as nn
import copy # 引入 copy
from Brain.DQN.lib import environment, common, model
import time






def test_grad_robust(model_template, dummy_input):
    # 使用 deepcopy 創建兩個完全獨立、相同權重的模型
    eager_model = copy.deepcopy(model_template)
    compiled_model_base = copy.deepcopy(model_template)
    
    # 編譯其中一個
    compiled_model = torch.compile(compiled_model_base)

    parameters_eager = {name: p for name, p in eager_model.named_parameters()}
    parameters_compiled = {name: p for name, p in compiled_model._orig_mod.named_parameters()}


    # --- 為了測試，將兩者都設置為 .eval() 模式以關閉 Dropout ---
    print("--- Running in .eval() mode to disable Dropout ---")
    eager_model.eval()
    compiled_model.eval()

    # --- 1. 先比較前向傳播的輸出 ---
    output_eager = eager_model(dummy_input)
    output_compiled = compiled_model(dummy_input)




    if torch.allclose(output_eager, output_compiled, atol=1e-5): # 放寬一點容忍度
        print("✅ Forward pass outputs are consistent!")
    else:
        print("❌ WARNING: Forward pass outputs are DIFFERENT!")
        diff = torch.linalg.norm(output_eager - output_compiled)
        print(f"   - Norm of difference: {diff.item()}")

    # --- 2. 比較梯度 ---
    eager_model.zero_grad()
    compiled_model.zero_grad()

    loss_eager = output_eager.sum()
    loss_eager.backward()
    grads_eager = {name: p.grad.clone() for name, p in eager_model.named_parameters()}
    eager_model.zero_grad()

    loss_compiled = output_compiled.sum()
    loss_compiled.backward()
    # 記得從 _orig_mod 獲取梯度
    grads_compiled = {name: p.grad.clone() for name, p in compiled_model._orig_mod.named_parameters()}
    compiled_model.zero_grad()




    # --- 比較 ---
    all_correct = True
    mismatched_layers = []
    mismatched_layers_data = []
    for name in grads_eager.keys():
        if not torch.allclose(grads_eager[name], grads_compiled[name], atol=1e-6):
            mismatched_layers.append(name)
            mismatched_layers_data.append((grads_eager[name], grads_compiled[name]))
            all_correct = False
    
    if all_correct:
        print("✅ All gradients are consistent in .eval() mode!")
    else:
        print(f"❌ Gradients mismatch in {len(mismatched_layers)} layers (in .eval() mode):")
        # 只印出前幾個不匹配的層，避免洗版
        for name in mismatched_layers[:5]:
            print(f"   - {name}")

        for data in mismatched_layers_data[:5]:
            print(f"   - {data[0]} vs {data[1]}")



# --- 執行新的測試 ---
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

dummy_data = torch.randn(32, 300, 12, device=DEVICE)

# 使用新的、更健壯的測試函式
test_grad_robust(net, dummy_data)
import torch
import torch.nn as nn
import copy
from Brain.DQN.lib import environment, common, model
import time



def compare_tensors(tensor_a, tensor_b, name, atol=1e-6, rtol=1e-6):
    """比較兩個張量，並輸出差異統計資訊"""
    diff = (tensor_a - tensor_b).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    relative_diff = (diff / tensor_a.abs().clamp(min=1e-8)).max().item()

    print(f"\n--- Comparing {name} ---")
    print(f"Max abs diff: {max_abs_diff:.6e}")
    print(f"Mean abs diff: {mean_abs_diff:.6e}")
    print(f"Max relative diff: {relative_diff:.6e}")

    if torch.allclose(tensor_a, tensor_b, atol=atol, rtol=rtol):
        print(f"✅ {name} tensors are consistent within tolerance.")
        return True
    else:
        print(f"❌ {name} tensors mismatch!")
        return False


def test_grad_robust(model_template, dummy_input):
    # 深拷貝兩個完全獨立的模型
    eager_model = copy.deepcopy(model_template)
    compiled_model_base = copy.deepcopy(model_template)
    
    # 編譯其中一個模型
    compiled_model = torch.compile(compiled_model_base)

    # 建立參數字典
    parameters_eager = {name: p for name, p in eager_model.named_parameters()}
    parameters_compiled = {name: p for name, p in compiled_model._orig_mod.named_parameters()}

    print("\n--- Running in .eval() mode to disable Dropout ---")
    eager_model.eval()
    compiled_model.eval()


    output_eager = eager_model(dummy_input)
    output_compiled = compiled_model(dummy_input)

    compare_tensors(output_eager, output_compiled, "Forward Output", atol=1e-4, rtol=1e-6)

    # 2. 反向傳播比較
    eager_model.zero_grad()
    compiled_model.zero_grad()

    loss_eager = output_eager.sum()
    loss_eager.backward()
    grads_eager = {name: p.grad.clone() for name, p in eager_model.named_parameters()}
    eager_model.zero_grad()

    loss_compiled = output_compiled.sum()
    loss_compiled.backward()
    grads_compiled = {name: p.grad.clone() for name, p in compiled_model._orig_mod.named_parameters()}
    compiled_model.zero_grad()

    # 比較每一層的梯度
    mismatched_layers = []
    for name in grads_eager.keys():
        print(f"\nChecking gradient for layer: {name}")
        is_close = compare_tensors(grads_eager[name], grads_compiled[name], f"Gradient: {name}", atol=1e-6, rtol=1e-6)
        if not is_close:
            mismatched_layers.append(name)

    if not mismatched_layers:
        print("\n✅ All gradients are consistent in .eval() mode!")
    else:
        print(f"\n❌ Gradients mismatch in {len(mismatched_layers)} layers:")
        for name in mismatched_layers[:5]:
            print(f"   - {name}")


# --- 測試執行 ---
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

# 執行比較測試
test_grad_robust(net, dummy_data)

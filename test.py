import numpy as np

def clip_grad_norm_(grads, max_norm=0.5):
    # grads 是一個 list 或 ndarray，代表不同參數的梯度
    # 我們先把它們疊起來視為一個大向量，計算整體范數
    flat_grads = np.concatenate([g.flatten() for g in grads])
    total_norm = np.linalg.norm(flat_grads)
    
    if total_norm > max_norm:
        # 計算縮放比例
        scale = max_norm / (total_norm + 1e-6)
        # 對每個梯度等比例縮放
        grads = [g * scale for g in grads]
    return grads

# 假設我們有兩個參數的梯度
g1 = np.array([10.0, 10.0])  # 兩個維度的梯度都很大
g2 = np.array([0.01, -0.01])

grads = [g1, g2]

print("Before clipping:")
for i, g in enumerate(grads):
    print(f"Gradient {i+1} norm: {np.linalg.norm(g)}")

# 執行梯度剪裁
grads_clipped = clip_grad_norm_(grads, max_norm=0.5)

print("\nAfter clipping:")
for i, g in enumerate(grads_clipped):
    print(f"Gradient {i+1} norm: {np.linalg.norm(g)}")

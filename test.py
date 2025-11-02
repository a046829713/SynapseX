from Brain.PPO2.lib.model import HybridMlpPolicy
import torch

def test_hybrid_mlp_policy():
    """
    測試 HybridMlpPolicy 模型的輸入和輸出是否符合預期。
    """
    print("--- 測試 HybridMlpPolicy ---")

    # 1. 定義超參數
    batch_size = 32
    ob_dim = 128          # 假設觀測空間扁平化後有 128 維
    discrete_ac_dim = 3   # 假設有 3 個離散動作 (例如：買, 賣, 持有)
    continuous_ac_dim = 1 # 假設有 1 個連續動作 (例如：下單比例)
    hid_size = 64
    num_hid_layers = 2

    print(f"Batch Size: {batch_size}")
    print(f"Observation Dim: {ob_dim}")
    print(f"Discrete Action Dim: {discrete_ac_dim}")
    print(f"Continuous Action Dim: {continuous_ac_dim}")

    # 2. 實例化模型
    model = HybridMlpPolicy(
        ob_dim=ob_dim,
        discrete_ac_dim=discrete_ac_dim,
        continuous_ac_dim=continuous_ac_dim,
        hid_size=hid_size,
        num_hid_layers=num_hid_layers
    )
    print("\n模型已成功建立。")

    # 3. 建立假的輸入張量 (扁平化的觀測值)
    flat_obs = torch.randn(batch_size, ob_dim)
    print(f"輸入觀測張量 (flat_obs) 的形狀: {flat_obs.shape}")

    # 4. 執行前向傳播
    discrete_logits, continuous_mu, continuous_log_std, value = model(flat_obs)
    print("\n模型前向傳播完成，輸出形狀如下：")
    print(f"  - 離散動作 Logits: {discrete_logits.shape}")
    print(f"  - 連續動作 Mu: {continuous_mu.shape}")
    print(f"  - 連續動作 Log_Std: {continuous_log_std.shape}")
    print(f"  - 價值 (Value): {value.shape}")

    # 5. 驗證輸出形狀
    assert discrete_logits.shape == (batch_size, discrete_ac_dim)
    assert continuous_mu.shape == (batch_size, continuous_ac_dim)
    assert continuous_log_std.shape == (continuous_ac_dim,)
    assert value.shape == (batch_size,)
    print("\n所有輸出形狀驗證成功！")
    print("--- 測試結束 ---\n")

if __name__ == "__main__":
    test_hybrid_mlp_policy()

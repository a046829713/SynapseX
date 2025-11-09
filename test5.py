import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# 假設您的 SineActivation, DAIN_Layer, MixerModel 已經被導入
# from your_modules import SineActivation, DAIN_Layer, MixerModel 

class (nn.Module):
    def __init__(self,
                 d_model: int,
                 nlayers: int,
                 num_actions: int,
                 time_features_in: int,
                 time_features_out: int = 32,
                 seq_dim: int = 300,
                 dropout: float = 0.1,
                 hidden_size: int = 96,
                 mode='full',
                 ssm_cfg: Optional[dict] = None,
                 moe_cfg: Optional[dict] = None,
=

                ):

        super().__init__()
        self.hidden_size = hidden_size
        self.seq_dim = seq_dim
        





 



# 假設你從 batch 中取出了：
# state, time_state, action, reward, next_state, next_time_state, done
# 以及 "想像的真實值" (例如未來 20 根 K 線的真實波動率)
# imagined_ground_truth [B, num_imagined_features]

# --- 計算 Q-values 和想像的預測 ---
# q_values 是 Q(s, a)
# imagined_preds 是想像路徑的預測
q_values, aux_loss, imagined_preds = model(state, time_state)
q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

# --- 計算 next Q-values (來自 Target Network) ---
with torch.no_grad():
    # 注意：Target Network 也返回 3 個值，但我們只關心 q_values
    next_q_values_target, _, _ = target_model(next_state, next_time_state)
    next_q_value = next_q_values_target.max(1)[0]
    target_q_value = reward + (1 - done) * gamma * next_q_value

# --- 1. 計算 RL 損失 ---
rl_loss = nn.MSELoss()(q_values, target_q_value)

# --- 2. 計算 "想像損失" ---
# imagined_ground_truth 是您在準備數據時計算好的 "未來真實值"
# 這就是 Model-Based 部分的監督訊號
imagination_loss = nn.MSELoss()(imagined_preds, imagined_ground_truth)

# --- 3. 總損失 ---
# 您需要設定一個權重來平衡這兩種損失
imag_loss_weight = 0.5 
total_loss = rl_loss + (imag_loss_weight * imagination_loss) + (aux_loss_weight * aux_loss)

# --- 反向傳播 ---
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
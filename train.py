








def ppo_update(policy_value_net, optimizer, transitions, clip_eps=0.2, epochs=10, batch_size=64):
    """
        this Version implement PPO Clip

        Loss function:
            L(\theta) = L^{\mathrm{CLIP}}(\theta) + c_{1}\,L^{\mathrm{VF}}(\theta) - c_{2}\,L^{\mathrm{H}}(\theta)

            C1(float):0.5 
            C2(float):0.01

        L Clip LaTex:
            L^{\mathrm{CLIP}}(\theta)
                = \hat{\mathbb{E}}_t\!\Bigl[
                    \min\Bigl(
                        r_t(\theta)\,\hat{A}_t,\;
                        \operatorname{clip}\!\bigl(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon\bigr)\,\hat{A}_t
                    \Bigr)
                \Bigr]

        to use in importance sampling.
        Ratio:
            r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
    """
    # 轉為 Tensor
    states      = torch.stack([torch.as_tensor(tr.state,  dtype=torch.float32) for tr in transitions]).to(device)
    actions     = torch.as_tensor([tr.action  for tr in transitions], device=device)
    old_logps   = torch.as_tensor([tr.logp    for tr in transitions], device=device)
    returns     = torch.as_tensor([tr.reward  for tr in transitions], device=device)
    advantages  = torch.as_tensor([tr.value   for tr in transitions], device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = torch.utils.data.TensorDataset(states, actions, old_logps, returns, advantages)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for s, a, old_lp, ret, adv in loader:
            logp_new, ent , value = policy_value_net.evaluate_actions(s, a)
            entropy = ent.mean()

            # policy ratio
            ratio = torch.exp(logp_new - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = nn.functional.mse_loss(value.view(-1), ret)


            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def save_model(model: torch.nn.Module,
               filepath: str = 'model.pt'):
    """
    儲存模型和（可選）優化器的狀態。

    參數：
    - model: 要儲存的 torch.nn.Module    
    - filepath: 檔案路徑（含檔名），例如 'ppo_cartpole.pt'

    儲存內容：
    - 'model_state_dict': model.state_dict()
    """
    # 準備要儲存的 dict
    checkpoint = {
        'model_state_dict': model.state_dict()
    }

    # 確保資料夾存在
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    # 儲存
    torch.save(checkpoint, filepath)
    print(f">>> Saved checkpoint to '{filepath}'")

# 範例使用：
# save_model(net, optimizer, 'ppo_cartpole.pth', epoch=100, extra={'reward_mean': 195.0})

# "LunarLander-v3"
# 'CartPole-v1'

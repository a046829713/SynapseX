import torch
import torch.optim as optim

class opitmizar():
    def __init__(self,net,learning_rate:float,lambda_l2:float):
        """
            For get Optimizar by myself.
        """
        self.net = net
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2
        self._prepare_optimizer()

    def get_optimizer(self):
        return self.optimizer

    def _prepare_optimizer(self, base_lr=1e-4):
        """
        建立 Adam 優化器，以下功能：
        1. `dean` 的三個特殊層 (`mean_layer`, `scaling_layer`, `gating_layer`) 使用獨立的學習率且不做 weight decay。
        2. `LayerNorm` 層和所有 `bias` 參數不做 weight decay。
        3. 其餘參數正常做 weight decay。
        """
        # 存放不同參數組
        decay_params = []
        no_decay_params = []
        
        # 獲取 dean 特殊層的參數 ID，以便後續排除
        dean_params_ids = set()
        if hasattr(self.net, 'dean'):
            dean_params_ids.update(id(p) for p in self.net.dean.parameters())

        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue

            # 如果是 dean 層的參數，跳過，因為它們會被單獨處理
            if id(param) in dean_params_ids:
                continue

            # LayerNorm 層和 bias 不做 weight decay
            # 透過 name 來判斷，比 isinstance 更可靠
            if "norm" in name or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # 建立參數組
        param_groups = [
            {
                'params': decay_params,
                'lr': self.learning_rate,
                'weight_decay': self.lambda_l2
            },
            {
                'params': no_decay_params,
                'lr': self.learning_rate,
                'weight_decay': 0.0
            }
        ]

        # 為 dean 的特殊層添加獨立的參數組
        if hasattr(self.net, 'dean'):
            param_groups.extend([
                {'params': list(self.net.dean.mean_layer.parameters()),
                 'lr': base_lr * self.net.dean.mean_lr, 'weight_decay': 0.0},
                {'params': list(self.net.dean.scaling_layer.parameters()),
                 'lr': base_lr * self.net.dean.scale_lr, 'weight_decay': 0.0},
                {'params': list(self.net.dean.gating_layer.parameters()),
                 'lr': base_lr * self.net.dean.gate_lr, 'weight_decay': 0.0},
            ])

        # 用 Adam 建立優化器
        self.optimizer = optim.AdamW(param_groups)
        
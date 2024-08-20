
import torch.nn as nn
import torch.nn.functional as F
import torch
import time

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 初始化位置編碼矩陣
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 雖然和幕運算不同但是數值更加穩定(ChatGpt說的XD)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)


        # 將位置編碼矩陣註冊為緩衝區，這樣它不會參與反向傳播
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 將位置編碼添加到輸入張量上
        x = x + self.pe[:x.size(1), :]
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout, batch_first=False):
        """
            此網絡並非原始架構，有將norm的順序顛倒。

        Args:
            input_dim (int): 輸入特徵的維度。
            num_heads (int): 注意力機制中的頭數。
            ff_dim (int): 前向全連接網絡的內部維度。
            dropout (float): Dropout 的概率。
            batch_first (bool): 如果为True，输入和输出张量形状应为 (batch, seq, feature)。默认False (seq, batch, feature)。

            Q : Why can't Dropout layers be shared?
            A : 因為 Dropout 會在每次前向傳播中隨機遮蔽不同的神經元，
            所以應用在不同的地方需要使用不同的 Dropout 層。每個 Dropout 層的遮蔽模式是隨機產生的，這樣才能有效地增加正則化效果並防止過擬合。

            Q : Why does the dropout change the original values?
            A :
                1.遮蔽神經元：Dropout 會隨機選擇一些神經元的輸出值設為零，這就是遮蔽神經元的過程。在訓練過程中，這有助於防止模型過度依賴某些特定的神經元，從而提高模型的泛化能力。
                2.縮放未被遮蔽的神經元：為了保持整體輸出值的期望不變，未被遮蔽的神經元的輸出值會按遮蔽概率 p 的倒數進行縮放。例如，若 Dropout 的遮蔽概率為 0.5，則未被遮蔽的神經元的輸出值會乘以 2（即 1/(1−0.5)1/(1−0.5)）。

                
            (尚未釐清)
            Q : how do you set parameter of MultiheadAttention ?
            A :
                embed_dim 可以設置為 input_dim 的倍數，並且能被 num_heads 整除。因此，embed_dim 可以設置為 120。這樣，每個注意力頭的嵌入維度會是 embed_dim / num_heads，即 120 / 10 = 12。

            (官方)
            self attention is being computed (i.e., query, key, and value are the same tensor).

            num_heads – Number of parallel attention heads. Note that embed_dim will be split across num_heads (i.e. each head will have dimension embed_dim // num_heads).
        """
        super(TransformerEncoderLayer, self).__init__()
        self.pos_encoder = PositionalEncoding(6, 300)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
        
        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.batch_first = batch_first

    def forward(self, src):
        src = self.pos_encoder(src)

        if not(self.batch_first):
            # Transpose src from (batch, seq, feature) to (seq, batch, feature) for self_attn
            src = src.transpose(0, 1)

        
        # Multi-head Self-Attention
        src2 = self.norm1(src)
        # attn_output, attn_output_weights = multihead_attn(query, key, value)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)

        # Feed-Forward Network
        src2 = self.norm2(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class TransformerDuelingModel(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_trans_blocks, num_actions, hidden_size, dropout=0):
        """
        初始化 Dueling Transformer 模型。
        Args:
            input_dim (int): 每个序列元素的特征维度。            
            num_heads (int): 注意力机制中头的数量。
            ff_dim (int): 前向全连接网络的内部维度。
            num_trans_blocks (int): Transformer 编码器块的数量。
            num_actions (int): 动作空间的大小，即输出层的维度。
            hidden_size (int): 隐藏层大小。
            dropout (float): Transformer 编码器中 Dropout 的比例。
        """
        super(TransformerDuelingModel, self).__init__()
        self.encoder_stack = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim, num_heads, ff_dim, dropout,batch_first=True)
            for _ in range(num_trans_blocks)
        ])

        # 状态值网络
        self.fc_val = nn.Sequential(
            nn.Linear(input_dim * 300, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # 优势网络
        self.fc_adv = nn.Sequential(
            nn.Linear(input_dim * 300, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        for encoder in self.encoder_stack:
            x = encoder(x)

        # 將 x 展平
        x_flat = x.view(x.size(0), -1)
        val = self.fc_val(x_flat)
        adv = self.fc_adv(x_flat)
        # 使用优势和值函数计算 Q 值
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values

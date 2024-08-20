import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils.Debug_tool import debug

# class DQNConv1D(nn.Module):
#     def __init__(self, shape, actions_n):
#         """捲積網路的參數詳解
#             in_channels = 特徵的數量
#             out_channels 出來的特徵數量(如果有100張,可以想像就是有100張小圖)
#             kernel_size (濾波器的大小)
#             stride = (移動步長)
#             padding =補0

#             # 這邊指的是input 是指輸入的特徵,口訣就是批次大小,通道數,數據長度

#             input_data 是三维的，因为卷积层（nn.Conv1d）的输入需要具有特定的形状，以便正确地进行卷积操作。这三个维度分别代表：

#             批次大小（Batch Size）：

#             代表一次输入到网络中的样本数量。通过一次处理多个样本（一个批次），网络可以加速训练，并且可以得到梯度的更稳定估计。

#             通道数（Channels）：

#             对于图像，通道通常代表颜色通道（例如，RGB）。在其他类型的数据中，通道可以代表不同类型的输入特征。
#             在一维卷积中，通道数可能对应于输入数据的不同特征。

#             数据长度（Data Length）：

#             这是输入数据的实际长度。对于图像，这会是图像的宽度和高度。对于一维数据，如时间序列，这会是序列的长度。
#         Args:
#             shape (_type_): _description_
#             actions_n (_type_): _description_
#         """
#         super(DQNConv1D, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv1d(shape[0], 128, kernel_size=5),
#             nn.ReLU(),
#             nn.Conv1d(128, 128, 5),
#             nn.ReLU(),
#         )

#         out_size = self._get_conv_out(shape)

#         self.fc_val = nn.Sequential(
#             nn.Linear(out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1)
#         )

#         self.fc_adv = nn.Sequential(
#             nn.Linear(out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, actions_n)
#         )

#     def forward(self, x):
#         # 要丟進去一般網絡之前要先flatten
#         conv_out = self.conv(x).view(x.size()[0], -1) #執行flatten的動作


#         val = self.fc_val(conv_out)
#         adv = self.fc_adv(conv_out)
#         return val + adv - adv.mean(dim=1, keepdim=True)

#     def _get_conv_out(self, shape):
#         """用來計算flatten後的參數個數

#         Args:
#             shape (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         o = self.conv(torch.zeros(1, *shape))
#         return int(np.prod(o.size()))


# 還可以加入dropout 和 批次正則
class DQNConv1D_Large(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1D_Large, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # Max pooling layer added here
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Another max pooling layer

            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # And another

            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Final max pooling layer
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, actions_n)
        )

    def forward(self, x):
        # Flatten the tensor after conv layers
        conv_out = self.conv(x).view(x.size()[0], -1)

        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)

    def _get_conv_out(self, shape):
        with torch.no_grad():  # To ensure it doesn't track history in autograd
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, add_feature_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.add_feature_size = add_feature_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        self.fc_val = nn.Sequential(
            nn.Linear(hidden_size + add_feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(hidden_size + add_feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size)
        )
        
    def mix_state(self, x, current_info):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)        
        
        out, _ = self.lstm(x, (h0, c0))
        finance_output = torch.cat((out[:, -1, :], current_info),dim=1)
        val = self.fc_val(finance_output)
        adv = self.fc_adv(finance_output)        
        return val + adv - adv.mean(dim=1, keepdim=True)
    
    def forward(self, x, current_info):        
        return self.mix_state(x,current_info)
    



# import torch
# import torch.nn as nn

# class SimpleLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, add_feature_size, output_size, num_layers=1):
#         super(SimpleLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.add_feature_size = add_feature_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

#         # 這裡設置了一個全連接層網絡，用於將LSTM輸出進行轉換
#         self.transform = nn.Linear(hidden_size + add_feature_size, 1024)

#         # 添加了一個殘差塊
#         self.residual_block = nn.Sequential(
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024)
#         )

#         # 獲得最終值
#         self.fc_val = nn.Linear(1024, 1)
#         self.fc_adv = nn.Linear(1024, output_size)

#     def mix_state(self, x, current_info):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)        
#         out, _ = self.lstm(x, (h0, c0))
#         finance_output = torch.cat((out[:, -1, :], current_info),dim=1)
        
#         # 應用初始的轉換
#         transformed_output = self.transform(finance_output)
        
#         # 應用殘差塊
#         residual_output = self.residual_block(transformed_output) + transformed_output
        
#         # 計算最終輸出
#         val = self.fc_val(residual_output)
#         adv = self.fc_adv(residual_output)
        
#         return val + adv - adv.mean(dim=1, keepdim=True)
    
#     def forward(self, x, current_info):        
#         return self.mix_state(x, current_info)




import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Parameter(torch.rand(max_length, embed_size))

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding[:seq_length])

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Parameter(torch.rand(max_length, embed_size))

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding[:seq_length])

        for layer in self.layers:
            x = layer(x, enc_out, x, trg_mask)

        out = self.fc_out(x)
        return out

# Assume some device (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example of model initialization
# model = TransformerBlock(embed_size=256, heads=8, dropout=0.1, forward_expansion=4)
# The rest of the model can be assembled similarly

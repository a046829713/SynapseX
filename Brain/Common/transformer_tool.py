from typing import Optional, Union, Callable
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.transformer import MultiheadAttention
from torch.nn.modules.transformer import _get_activation_fn
import math

class TransformerEncoderLayer(nn.Module):
    r"""
    TransformerEncoderLayer 由自注意力和前饋神經網絡組成。
    這個標準的編碼器層基於論文 "Attention Is All You Need"。該論文由 Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N Gomez、Lukasz Kaiser 和 Illia Polosukhin 於 2017 年發表在 Advances in Neural Information Processing Systems 上，頁碼 6000-6010。用戶可以在應用過程中修改或以不同方式實現這個編碼器層。
    TransformerEncoderLayer 可以處理傳統的 torch.tensor 輸入，也可以處理嵌套張量（Nested Tensor）輸入。預期派生類別應同樣接受這兩種輸入格式。（目前 TransformerEncoderLayer 在嵌套張量處於原型狀態時，並不支持所有組合的輸入。）
    如果您正在實現自定義層，可以從 Module 或 TransformerEncoderLayer 類別繼承。如果您的自定義層支持 torch.Tensors 和嵌套張量輸入，則使其實現派生自 TransformerEncoderLayer。如果您的自定義層僅支持 torch.Tensor 輸入，則從 Module 繼承其實現。

    # 我採用正規傳統的torch.tensor ,所以我直接繼承nn.Module

    參數:
        d_model: 輸入中預期的特徵數量（必填）。
        nhead: 多頭注意力模型中的頭數（必填）。
        dim_feedforward: 前饋神經網絡模型的維度（默認值=2048）。
        dropout: dropout 的值（默認值=0.1）。
        activation: 中間層的激活函數，可以是字串（"relu" 或 "gelu"）或單一可調用對象。默認值: relu
        layer_norm_eps: 層正則化組件中的 eps 值（默認值=1e-5）。
        batch_first: 如果為 ``True``，則輸入和輸出張量的格式為 (batch, seq, feature)。默認值: ``False`` (seq, batch, feature)。
        bias: 如果設置為 ``False``，則 ``Linear`` 和 ``LayerNorm`` 層不會學習加性偏差。默認值: ``True``。

    示例::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    或者當 ``batch_first`` 為 ``True`` 時：
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    快速路徑：
        當滿足以下所有條件時，forward() 將使用一種特殊的優化實現，該實現描述於
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ ：

        - 自動微分被禁用（使用 ``torch.inference_mode`` 或 ``torch.no_grad``）或沒有張量參數 ``requires_grad``
        - 訓練被禁用（使用 ``.eval()``）
        - batch_first 為 ``True`` 且輸入是批次格式（即 ``src.dim() == 3``）
        - 激活函數是以下之一：``"relu"``, ``"gelu"``, ``torch.functional.relu``, 或 ``torch.functional.gelu``
        - 至多傳遞 ``src_mask`` 和 ``src_key_padding_mask`` 之一
        - 如果 src 是 `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_，則不傳遞 ``src_mask`` 和 ``src_key_padding_mask``
        - 兩個 ``LayerNorm`` 實例具有一致的 ``eps`` 值（這將自然成立，除非調用者手動修改其中之一而未修改另一個）

        如果使用優化實現，可以為 ``src`` 傳遞 `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_，以比使用填充掩碼更高效地表示填充。在這種情況下，將返回 `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_，並且可以預期相較於輸入中填充的比例會有額外的加速。

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
        https://arxiv.org/abs/2205.14135

    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 bias: bool = True, # 不須考慮
                 device=None,
                 dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}


        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(
            d_model, dim_feedforward, bias=bias, **factory_kwargs)
        
        self.dropout = nn.Dropout(dropout)
        
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, bias=bias, **factory_kwargs)

        
        self.dropout1 = nn.Dropout(dropout)
        
        self.dropout2 = nn.Dropout(dropout)

        # ReZero 參數
        self.alpha = nn.Parameter(torch.zeros(1))


        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        
        
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        # out_put = None
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # ReZero 機制應用於自注意力塊
        src2 = self._sa_block(src, src_mask, src_key_padding_mask, is_causal=is_causal)
        src = src + self.alpha * src2

        # ReZero 機制應用於前饋神經網絡塊
        src2 = self._ff_block(src)
        src = src + self.alpha * src2

        return src

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

        # Calculate positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register pe as a buffer to make it part of the model state (not a trainable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Add positional encoding to the input embeddings        
        x = x + self.pe[:x.size(0)]
        
        # Apply dropout and return the result
        return self.dropout(x)

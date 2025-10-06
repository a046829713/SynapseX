from soft_moe_pytorch import SoftMoE,CausalSoftMoE
import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, einsum, Tensor

from einops import rearrange, pack, unpack

from soft_moe_pytorch.distributed import (
    AllGather,
    split_by_rank,
    gather_sizes,
    has_only_one_value
)
import torch



moe = CausalSoftMoE(
    dim = 12,         
    num_experts = 16    
)

x = torch.randn(1, 300, 12)

out = moe(x) + x # (1, 1024, 512) - add in a transformer in place of a feedforward at a certain layer (here showing the residual too)
print(out)
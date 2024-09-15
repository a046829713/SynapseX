import torch.distributed as dist
import torch

def setup_distributed():
    dist.init_process_group(backend='nccl')  # 使用NCCL後端
    print(dist.get_rank())
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup_distributed():
    dist.destroy_process_group()
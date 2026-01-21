import torch


x = torch.randn([1,10,3])
print(x)
print(x.size())
print(torch.mean(x,dim = 1))
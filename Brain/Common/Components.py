import torch
import torch.nn as nn


def t2v(tau, f, w, b, w0, b0, arg=None):
    """
    Time2Vec helper function.
    Computes the Time2Vec embedding.
    """
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    
    v2 = torch.matmul(tau, w0) + b0
    output = torch.cat([v1, v2], -1)
    return output


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)

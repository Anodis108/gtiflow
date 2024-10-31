import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12) -> None:
        super(LayerNorm).__init__()
        self .gamma = nn.Parameter(torch.ones(d_model)) # Tạo tham số ban đầu
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 
        
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
    
import torch
from torch import nn
import math

class PositionalEncoding_aHung(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Khởi tạo lớp Dropout
        self.dropout = nn.Dropout(p=0.1)
        # Khởi tạo 1 ma trận position encoder có cùng kích thước với đầu vào d_model
        pe = torch.zeros(max_len, d_model)
        # Khởi tạo vị trí (size = [max_len, 1])
        position = torch.arange(0, d_model, dtype=torch.float).float().unsqueeze(1)
        # Khởi tạo công thức tính W_k
        div_term = torch.exp( torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Khởi tạo công thức tính f(t)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Thêm 1 chiều với dim bằng 0 cho Tensor 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Kết quả quá trình Position encoder
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
from torch import nn

class EncoderLayer_aHung(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super().__init__()
        #Khởi tạo lớp multi-head attention tương ứng với n_heads
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads)
        # Khởi tạo norm layers
        self.norm1 = nn.LayerNorm(d_model)
        # khởi tạo feed forward layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, d_model),
        )
        # Khởi tọa lóp Norm thứ 2
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self,x):
        # Đưa input vào lớp Add & Norm và tính toán qua lớp multi-head attention
        x = self.norm1(x)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + attn_output
        # tiếp tục đưa x qua lớp Norm và tính toán output của multi head attention layer vào khối feed forward
        x = self.norm2(x)
        ff_output = self.feedforward(x)
        x = x + ff_output
        # Thu được đầu ra của khối encoder
        return x

# Decoderlayers
class Decoderlayer_aHung(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Khởi tạo lớp multi_head attention tương ứng với n_heads
        self.multihead_attn1 = nn.MultiheadAttention(d_model, n_heads)
        # Khởi tạo Nỏm layers
        self.norm1 = nn.LayerNorm(d_model)
        # Khởi tạo lớp multi-head attention thứ 2 tương đương với n-heads
        self.multihead_attn2 = nn.MultiheadAttention(d_model, n_heads)
        # Khởi tạo Norm layers thứ 2
        self.norm2 = nn.LayerNorm(d_model)
        # Khởi tạo feed forward layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, d_model),
        )
        # Khởi tạo lớp Norm thứ 3
        self.norm3 = nn.LayerNorm(d_model)
        
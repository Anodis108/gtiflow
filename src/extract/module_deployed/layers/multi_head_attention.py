from torch import nn

from module_deployed.layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model) # Thực hiện đại số tuyến tính y=xA^T+b.
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)     
        self.w_concat = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 2. split tensor by number of heads
        q, k, v = self.split(q),  self.split(k),  self.split(v)
        
        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)
        
        # 5. visualize attention map
        # TODO : we should implement visualization => để làm sau
        
        return out
        
    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2) # Thay đổi kích thước của tensor thành [batch_size, length, n_head, d_tensor] rồi chuyển vị để trở thành [batch_size, n_head, length, d_tensor].
        # it is similar with group convolution (split by number of heads)
        
        return tensor
        
    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length , d_tensor = tensor.size() # Vì bê trên có transpose(1, 2) nên thay đổi vị trí đầu vào 2 biến head và lenght
        d_model = head * d_tensor
        
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
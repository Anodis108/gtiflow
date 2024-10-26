import torch
from torch import nn

from model.encoder import Encoder
from model.decoder import Decoder

class Transformer(nn.Module):
    
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device) -> None:
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) # Trc đó, càn thêm padding để kích thước các token = nhau
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device) # Tạo ma trận tất cả bằng 1, sd trill để giữ lại phần tam giác dưới, còn trên để = 0
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
        #  VD trg = [[4, 7, 2, 0, 0]]
        # -> trg_pad_mask = trg != trg_pad_idx → [[True, True, True, False, False]]
        #   trg_pad_mask =
        # [[[[True],
        #    [True],
        #    [True],
        #    [False],
        #    [False]]]]
        # trg_sub_mask =
        # [[1, 0, 0, 0, 0],
        #  [1, 1, 0, 0, 0],
        #  [1, 1, 1, 0, 0],
        #  [1, 1, 1, 1, 0],
        #  [1, 1, 1, 1, 1]]
        # trg_mask =trg_pad_mask & trg_sub_mask =
        # [[[[True,  False, False, False, False],
            # [True,  True,  False, False, False],
            # [True,  True,  True,  False, False],
            # [False, False, False, False, False],
            # [False, False, False, False, False]]]]
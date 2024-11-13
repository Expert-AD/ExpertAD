import torch
import torch.nn as nn
import torch.nn.functional as F
from .sliding_window_attention import SlidingWindowAttention
from .block_sparse_attention import BlockSparseAttention
from .global_sparse_attention import GlobalSparseAttention
from .moa_attention import MoaAttention

class InteractionDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, window_size=3, block_size=3, topk=4, activation="relu", layer_norm_eps=1e-5, batch_first=False, norm_first=False, device=None, dtype=None):
        super(InteractionDecoder, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs).to(device)
        self.sliding_window_attention = SlidingWindowAttention(
            embed_dim=d_model,
            num_heads=nhead,
            window_size=window_size 
        ).to(device)
        self.block_sparse_attention = BlockSparseAttention(
            embed_dim=d_model,
            num_heads=nhead,
            block_size=block_size
        ).to(device)
        self.global_sparse_attention = GlobalSparseAttention(
            embed_dim=d_model,
            num_heads=nhead,
            topk=topk
        ).to(device)
        self.moa_attention = MoaAttention(
            embed_dim=d_model,
            num_heads=nhead,
        ).to(device)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs).to(device)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs).to(device)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs).to(device)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs).to(device)
        self.dropout1 = nn.Dropout(dropout).to(device)
        self.dropout2 = nn.Dropout(dropout).to(device)
        self.dropout3 = nn.Dropout(dropout).to(device)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, pattern, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        device = x.device
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            attn = self._mha_block(self.norm2(x), memory, pattern,  memory_mask, memory_key_padding_mask)
            x = x + attn
            x = x + self._ff_block(self.norm3(x))
        else:
            self.norm1.to(device)
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            attn = self._mha_block(x, memory, pattern, memory_mask, memory_key_padding_mask)
            self.norm2.to(device)
            x = self.norm2(x + attn)
            self.norm3.to(device)
            x = self.norm3(x + self._ff_block(x))
        return x

    # Self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        device = x.device
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device)
        x = x.to(device)
        self.self_attn.to(device)
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0].to(device)
        return self.dropout1(x)
    # Multihead attention block
    def _mha_block(self, x, mem , pattern, attn_mask, key_padding_mask):
        device = x.device
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device) 
        mem = mem.to(device)      
        if pattern == "sliding_window":
            # Use sliding window attention
            self.sliding_window_attention.to(device)
            x = self.sliding_window_attention(x, mem, mem).to(device) 
        elif pattern == "block_sparse":
            # Use block sparse attention
            self.block_sparse_attention.to(device)
            x = self.block_sparse_attention(x, mem, mem).to(device) 
        elif pattern == "global_sparse":
            # Use global sparse attention
            self.global_sparse_attention.to(device)
            x = self.global_sparse_attention(x, mem, mem).to(device) 
        # elif pattern == "moa_attention":
        #     # Use moa attention
        #     x = self.moa_attention(x, mem, mem)
        else:
            raise ValueError(f"Unknown pattern type: {pattern}")
        return self.dropout2(x)

    # Feed forward block
    def _ff_block(self, x):
        device = x.device
        self.linear1.to(device)
        self.linear2.to(device)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
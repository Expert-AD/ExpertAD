import torch
import torch.nn as nn
import math

class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super(SlidingWindowAttention, self).__init__()
        
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.cached_mask = None

        # Linear layers to project the input embeddings to query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, q, k, v):
        batch_size, seq_len, embed_dim = q.size()
        
        Q = self.q_proj(q).contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_output = self.sliding_window_attention(Q, K, V, self.window_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attention_output)

    def create_sliding_window_mask(self, seq_len, window_size):
        # Create a mask using band matrix with diagonal window
        mask = torch.ones(seq_len, seq_len, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=-window_size) * torch.tril(mask, diagonal=window_size)
        return mask

    def sliding_window_attention(self, Q, K, V, window_size):
        batch_size, num_heads, seq_len, head_dim = Q.size()
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Create sliding window mask
        if self.cached_mask is None or self.cached_mask.size(0) != seq_len:
            self.cached_mask = self.create_sliding_window_mask(seq_len, self.window_size).to(Q.device)
    
        # Use cached mask instead of recomputing
        mask = self.cached_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, seq_len, seq_len)
        
        # Apply the mask, setting the non-window region to negative infinity
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        
        output = torch.matmul(attention_probs, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        return output
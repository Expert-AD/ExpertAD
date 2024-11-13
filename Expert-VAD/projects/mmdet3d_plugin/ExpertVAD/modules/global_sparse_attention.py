import torch
import torch.nn as nn
import math

class GlobalSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, topk):
        super(GlobalSparseAttention, self).__init__()
        
        self.num_heads = num_heads
        self.topk = topk  
        self.head_dim = embed_dim // num_heads

        # Linear layers to project the input embeddings to query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        batch_size, seq_len, embed_dim = q.size()
        
        # Project the inputs to query, key, value
        Q = self.q_proj(q).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)
        K = self.k_proj(k).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)
        V = self.v_proj(v).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)

        # Apply top-k sparse attention
        attention_output = self.top_k_attention(Q, K, V)
        attention_output = attention_output.view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attention_output)

    def top_k_attention(self, Q, K, V):
        batch_size, num_heads, seq_len, head_dim = Q.size()

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

        # Get top-k scores and indices
        topk_scores, topk_indices = torch.topk(attention_scores, self.topk, dim=-1)

        # Apply softmax to topk scores
        attention_probs = torch.nn.functional.softmax(topk_scores, dim=-1)

        # Gather selected values
        # Expand indices to gather the selected values
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
        selected_values = torch.gather(V.unsqueeze(2).expand(-1, -1, seq_len, -1, -1), -2, topk_indices_expanded)

        # Compute weighted sum of selected values
        output = torch.einsum('bhqk,bhqkd->bhqd', attention_probs, selected_values)

        return output
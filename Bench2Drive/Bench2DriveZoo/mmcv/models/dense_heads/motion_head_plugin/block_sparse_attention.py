import torch
import torch.nn as nn
import math


class BlockSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, num_global_blocks=1):
        super(BlockSparseAttention, self).__init__()
        
        self.num_heads = num_heads
        self.block_size = block_size
        self.num_global_blocks = num_global_blocks  
        self.head_dim = embed_dim // num_heads

        # Linear layers to project the input embeddings to query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, embed_dim = q.size()
        
        # Project the inputs to query, key, value
        Q = self.q_proj(q).contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply block sparse attention
        attention_output = self.block_sparse_attention(Q, K, V, self.block_size, self.num_global_blocks, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attention_output)

    def block_sparse_attention(self, Q, K, V, block_size: int, num_global_blocks: int, mask=None):
        batch_size, num_heads, seq_len, head_dim = Q.size()
        
        # Number of blocks
        num_blocks = (seq_len + block_size - 1) // block_size

        # Initialize output tensor
        output = torch.zeros_like(Q)

        # Block indices
        block_indices = torch.arange(num_blocks * block_size, device=Q.device).view(num_blocks, block_size)
        block_indices = block_indices.clamp(max=seq_len - 1)

        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, seq_len)

            # Local block attention (within the block)
            Q_block = Q[:, :, start_idx:end_idx, :].contiguous()
            K_block = K[:, :, start_idx:end_idx, :].contiguous()
            V_block = V[:, :, start_idx:end_idx, :].contiguous()

            # Compute local attention within the block
            local_attention_scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) / math.sqrt(head_dim)
            local_attention_probs = torch.nn.functional.softmax(local_attention_scores, dim=-1)
            block_output = torch.matmul(local_attention_probs, V_block)
            output[:, :, start_idx:end_idx, :] = block_output

            # Sparse cross-block attention with global blocks
            if num_global_blocks > 0:
                global_start_idx = max(0, i - num_global_blocks) * block_size
                global_end_idx = min(num_blocks, i + num_global_blocks + 1) * block_size

                K_cross_block = K[:, :, global_start_idx:global_end_idx, :].contiguous()
                V_cross_block = V[:, :, global_start_idx:global_end_idx, :].contiguous()

                # Compute cross-block attention
                cross_attention_scores = torch.matmul(Q_block, K_cross_block.transpose(-2, -1)) / math.sqrt(head_dim)
                cross_attention_probs = torch.nn.functional.softmax(cross_attention_scores, dim=-1)
                cross_block_output = torch.matmul(cross_attention_probs, V_cross_block)

                # Add sparse cross-block attention output
                output[:, :, start_idx:end_idx, :] += cross_block_output

        return output
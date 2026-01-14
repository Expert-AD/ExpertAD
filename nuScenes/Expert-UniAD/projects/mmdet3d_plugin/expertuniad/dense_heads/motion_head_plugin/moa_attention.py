import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


class MoaAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MoaAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projections for query, key, value
        self.q_proj = MoE(embed_dim, self.head_dim)

        # Dropout
        self.dropout_module = nn.Dropout(dropout)

        # Final output projection
        self.out_proj = nn.Linear(self.head_dim * num_heads, embed_dim)

    def forward(self, query, key, value):
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        # Project query, key, and value
        q = self.q_proj.map(query).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = value.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attn_weights = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.head_dim)

        # Softmax attention weights
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Compute attention output
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v).transpose(1, 2)
        attn_output = self.q_proj.reduce(attn_output)


        attn_output = attn_output.contiguous().view(bsz, tgt_len, self.num_heads * self.head_dim)

        return self.out_proj(attn_output)

class MoE(nn.Module):
    def __init__(self, input_size, head_size, num_experts=16, topk=8):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.experts = ParallelExperts(num_experts, input_size, head_size)
        self.output_experts = ParallelExperts(num_experts, head_size, input_size)
        self.expert_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.topk = topk

    def top_k_gating(self, x, sample_topk=4):
        logits = x @ self.expert_gate
        probs = torch.softmax(logits, dim=1)

        if self.training and (sample_topk > 0):
            _, top_km1_indices = probs.topk(self.topk - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.topk, dim=1)
        
        zeros = torch.zeros_like(probs, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        self.expert_size = (gates > 0).long().sum(0)

        top_k_gates = top_k_gates.flatten()
        top_k_experts = top_k_indices.flatten()
        
        nonzeros = top_k_gates.nonzero().squeeze(-1)
        top_k_experts_nonzero = top_k_experts[nonzeros]

        _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
        self.index_sorted_experts = nonzeros[_index_sorted_experts]
        self.batch_index = self.index_sorted_experts.div(self.topk, rounding_mode='trunc') 
        self.batch_gates = top_k_gates[self.index_sorted_experts]

        return gates

    def map(self, x, sample_topk=4):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        gates = self.top_k_gating(x, sample_topk=sample_topk)
        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)
        zeros = torch.zeros((bsz * length * self.topk, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.topk, -1)
        return y

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.contiguous().view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y

class ParallelLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, expert_size, weight, bias=None):
        output_list = []
        expert_size_list = expert_size.tolist()
        # print("expert_size_list:", expert_size_list)
        # print("input_size:", input.shape)
        input_list = list(input.split(expert_size_list, dim=0))      
        for i in range(weight.size(0)):
            if bias is not None:
                o_i = torch.mm(input_list[i], weight[i]) + bias[i]
            else:
                o_i = torch.mm(input_list[i], weight[i])
            output_list.append(o_i)
        output = torch.cat(output_list, dim=0)
        variables = (input, expert_size, weight, bias)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        input, expert_size, weight, bias = ctx.saved_tensors
        num_linears = weight.size(0)

        expert_size_list = expert_size.tolist()
        input_list = input.split(expert_size_list, dim=0)
        grad_list = grad_out.split(expert_size_list, dim=0)

        d_input_list = []
        for i in range(num_linears):
            d_input_list.append(torch.einsum('bi,ji->bj', grad_list[i], weight[i]))
        d_input = torch.cat(d_input_list, dim=0)

        d_weight_list = []
        for i in range(num_linears):
            d_weight_list.append(torch.einsum('bi,bj->ij', input_list[i], grad_list[i]))
        d_weight = torch.stack(d_weight_list, dim=0)

        if bias is not None:
            d_bias_list = []
            for i in range(num_linears):
                d_bias_list.append(grad_list[i].sum(0))
            d_bias = torch.stack(d_bias_list, dim=0)
        else:
            d_bias = None
        return d_input, None, d_weight, d_bias


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False):
        super().__init__()

        self.w = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        if bias:
            self.b = nn.Parameter(torch.zeros(num_experts, output_size))
        else:
            self.b = None

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2.0 / float(self.w.size(1) + self.w.size(2)))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(self.w, -a, a)

    def forward(self, inputs, expert_size):
        results = ParallelLinear.apply(inputs, expert_size, self.w, self.b)
        return results

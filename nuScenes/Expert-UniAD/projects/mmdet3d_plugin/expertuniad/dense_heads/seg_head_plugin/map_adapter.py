import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils.registry import Registry

class LearnedRouter(nn.Module):
    def __init__(self, input_dim, k, epsilon, T=20):
        super(LearnedRouter, self).__init__()
        self.w = nn.Parameter(torch.randn(input_dim, dtype= torch.float32))  
        self.k = k
        self.epsilon = epsilon
        self.T = T


    def soft_top_k(self, s):
        n = len(s)
        a = torch.tensor(0.0, device=s.device, dtype= torch.float32)
        b = torch.zeros(n, device=s.device, dtype= torch.float32)

        for t in range(self.T):
            max_value = torch.max((s + b) / self.epsilon) 
            adjusted_exp = torch.exp((s + b) / self.epsilon - max_value) 
            sum_exp = torch.sum(adjusted_exp)  
            log_sum_exp = torch.log(sum_exp) + max_value  
            a_prime = self.epsilon * torch.log(torch.tensor(self.k, dtype=torch.float32)) - self.epsilon * log_sum_exp
            b_prime = torch.minimum(-s - a_prime, torch.tensor(0.0,device=s.device, dtype= torch.float32))
            
            a = a_prime
            b = b_prime
        
        lambda_ = torch.exp((s + b + a) / self.epsilon)
        return lambda_

    def top_k(self, lambda_):
        top_k_mask = torch.zeros_like(lambda_)
        top_k_values, top_k_indices = torch.topk(lambda_, self.k)
        top_k_mask[top_k_indices] = 1
        return top_k_mask, top_k_indices

    def forward(self, x):
        s = x * self.w
        lambda_ = self.soft_top_k(s)
        top_k_mask, top_k_indices = self.top_k(lambda_)
        m = lambda_ * top_k_mask
        return m,  top_k_indices
  
class CoDA_layer(nn.Module):
    def __init__(self, input_dim, k,  epsilon, T=20):
        super(CoDA_layer, self).__init__()
        self.learned_router = LearnedRouter(input_dim, k, epsilon, T)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x): 
        device = x.device
        self.layer_norm = self.layer_norm.to(device)
        self.learned_router = self.learned_router.to(device)

        bev_h_w, B, C = x.size()
        x_reshaped = x.view(bev_h_w, C)
        global_feats = x_reshaped.mean(dim=0)
        global_feats = self.layer_norm(global_feats)
        m, top_k_indices = self.learned_router.forward(global_feats)

        return  m, top_k_indices

class Adapter(nn.Module):
    def __init__(self, input_dim = 256 , output_dim = 128, hidden_dim = 256, k=128, epsilon=0.01, T=50):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(k, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.coda_layer = CoDA_layer(input_dim,  k, epsilon, T)

    def forward(self, x):
        m, top_k_indices = self.coda_layer(x)
        x = x[:, :, top_k_indices]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, m, top_k_indices
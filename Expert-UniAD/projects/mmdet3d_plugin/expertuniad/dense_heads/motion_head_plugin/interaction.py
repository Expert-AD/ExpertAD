import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner.base_module import BaseModule
from concurrent.futures import ThreadPoolExecutor
from projects.mmdet3d_plugin.models.utils.functional import (
    norm_points,
    pos2posemb2d,
    features2posemb2d,
    trajectory_coordinate_transform
)
from .InteractionDecoder import InteractionDecoder
import time
from torch.cuda.amp import custom_fwd, custom_bwd
from mmcv.utils import build_from_cfg

from mmcv.utils.registry import Registry

INTERACTION = Registry('interaction')


@INTERACTION.register_module()
class IntentionInteraction(BaseModule):
    """
    Modeling the interaction between anchors
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerEncoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims * 2,
                                                                  batch_first=batch_first)

    def forward(self, query):
        B, A, P, D = query.shape
        # B, A, P, D -> B*A, P, D
        rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = rebatch_x.view(B, A, P, D)
        return out


@INTERACTION.register_module()
class TrackInteraction(BaseModule):
    """
    Modeling the interaction between the agents
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 sparse_att=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.sparse_att=sparse_att
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                          nhead=num_heads,
                                                          dropout=dropout,
                                                          dim_feedforward=embed_dims * 2,
                                                          batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None, pattern=None):
        '''
        query: context query (B, A, P, D)
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        mem = key.view(B * A, D).unsqueeze(1).expand(B * A, P, D)
        # B, A, P, D -> B*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem, self.sparse_att)
        query = query.view(B, A, P, D)
        return query


@INTERACTION.register_module()
class MapInteraction(BaseModule):
    """
    Modeling the interaction between the agents
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 sparse_att=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.sparse_att=sparse_att  
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                          nhead=num_heads,
                                                          dropout=dropout,
                                                          dim_feedforward=embed_dims * 2,
                                                          batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None, pattern=None):
        '''
        query: context query (B, A, P, D)
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # B, M, D = key.shape
        mem = key.mean(dim=1).unsqueeze(1).expand(-1, A * P, -1).contiguous().view(B * A, P, D)

        # B, A, P, D -> B*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem, self.sparse_att)
        query = query.view(B, A, P, D)
        return query


@INTERACTION.register_module()
class VelocityInteraction(BaseModule):
    """
    Modeling the interaction between the agents
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 sparse_att=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.sparse_att=sparse_att 
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                          nhead=num_heads,
                                                          dropout=dropout,
                                                          dim_feedforward=embed_dims * 2,
                                                          batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None, pattern=None):
        '''
        query: context query (B, A, P, D)
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        mem = key.expand(B * A, -1, -1)
        # B, A, P, D -> B*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem, self.sparse_att)
        query = query.view(B, A, P, D)
        return query


@INTERACTION.register_module()
class AccelerationInteraction(BaseModule):
    """
    Modeling the interaction between the agents
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 sparse_att=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.sparse_att=sparse_att
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                          nhead=num_heads,
                                                          dropout=dropout,
                                                          dim_feedforward=embed_dims * 2,
                                                          batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None, pattern=None):
        '''
        query: context query (B, A, P, D)
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        mem = key.expand(B * A, -1, -1)
        # B, A, P, D -> B*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem, self.sparse_att)
        query = query.view(B, A, P, D)
        return query


@INTERACTION.register_module()
class YawInteraction(BaseModule):
    """
    Modeling the interaction between the agents
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 sparse_att=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.sparse_att=sparse_att 
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                          nhead=num_heads,
                                                          dropout=dropout,
                                                          dim_feedforward=embed_dims * 2,
                                                          batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None, pattern=None):
        '''
        query: context query (B, A, P, D)
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        mem = key.expand(B * A, -1, -1)
        # B, A, P, D -> B*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem, self.sparse_att)
        query = query.view(B, A, P, D)
        return query


@INTERACTION.register_module()
class ReferencePointInteraction(BaseModule):
    """
    Modeling the interaction between the agents
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 sparse_att=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.sparse_att=sparse_att 
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                          nhead=num_heads,
                                                          dropout=dropout,
                                                          dim_feedforward=embed_dims * 2,
                                                          batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None, pattern=None):
        '''
        query: context query (B, A, P, D)
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        mem = key.view(B * A, P, D)
        # B, A, P, D -> B*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem, self.sparse_att)
        query = query.view(B, A, P, D)
        return query


@INTERACTION.register_module()
class CommandInteraction(BaseModule):
    """
    Modeling the interaction between the agents
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 sparse_att=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.sparse_att=sparse_att
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                          nhead=num_heads,
                                                          dropout=dropout,
                                                          dim_feedforward=embed_dims * 2,
                                                          batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None, pattern=None):
        '''
        query: context query (B, A, P, D)
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        mem = key.expand(B * A, -1, -1)
        # B, A, P, D -> B*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem, self.sparse_att)
        query = query.view(B, A, P, D)
        return query


@INTERACTION.register_module()
class BevInteraction(BaseModule):
    """
    Modeling the interaction with bird's eye view (BEV) embeddings
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 transformerlayers=None,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.transformerlayers = transformerlayers
        self.interaction_transformer = build_transformer_layer(transformerlayers)

    def forward(self, query, value, query_pos=None, bbox_results=None, reference_trajs=None, **kwargs):
        """
        Args:
            query (Tensor): The query embeddings.
            value (Tensor): The BEV embeddings.
            query_pos (Tensor): Positional encodings for the queries.
            bbox_results (dict or Tensor): Information about detected bounding boxes.
            reference_trajs (Tensor): The reference trajectories.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The output embeddings after interaction.
        """
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos

        # Flatten (B, A, P, D) -> (B*A, P, D) as per the requirement of the transformer decoder layer.
        query = torch.flatten(query, start_dim=0, end_dim=1)

        # Perform cross-attention using query from agents and key, value from BEV embeddings.
        output = self.interaction_transformer(query=query, value=value, query_pos=query_pos, bbox_results=bbox_results,
                                              reference_trajs=reference_trajs, **kwargs)

        # Reshape output back to (B, A, P, D).
        output = output.view(B, A, P, D)

        return output




class RoutingNetwork(nn.Module):
    def __init__(self, embed_dims, num_experts, k, cvloss=0.1, switchloss=0.1, zloss=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_experts = num_experts
        self.k = k
        self.cvloss = cvloss
        self.switchloss = switchloss
        self.zloss = zloss
        self.w_gate = nn.Parameter(torch.randn(embed_dims, num_experts), requires_grad=True)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def forward(self, query):
        B, A, P, D = query.size()
        query_flat = query.view(-1, D)

        logits = query_flat @ self.w_gate
        probs = torch.softmax(logits, dim=1)
        top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

        return top_k_gates, top_k_indices

@INTERACTION.register_module()
class MoATransformerInteraction(nn.Module):
    """
    Interaction layer with Mixture of Attention (MoA) architecture
    """

    def __init__(self,
                 num_experts=8,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 #transformerlayers=None,
                 k=4,
                 experts=None,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__()

        self.batch_first = batch_first
        self.k = k

        # gating
        self.routing_network = RoutingNetwork(embed_dims, num_experts, k)

        # Parallel_expert
        # self.experts = nn.ModuleList([
        #     TrackInteraction(embed_dims, num_heads, dropout, batch_first),
        #     MapInteraction(embed_dims, num_heads, dropout, batch_first),
        #     VelocityInteraction(embed_dims, num_heads, dropout, batch_first),
        #     AccelerationInteraction(embed_dims, num_heads, dropout, batch_first),
        #     YawInteraction(embed_dims, num_heads, dropout, batch_first),
        #     ReferencePointInteraction(embed_dims, num_heads, dropout, batch_first),
        #     CommandInteraction(embed_dims, num_heads, dropout, batch_first),
        #     BevInteraction(embed_dims, num_heads, dropout, batch_first, transformerlayers)
        # ])
        self.experts = nn.ModuleList([build_from_cfg(expert_cfg, INTERACTION) for expert_cfg in experts])

        self.num_experts = num_experts

        self.layer_norm = nn.LayerNorm(embed_dims)

        self.out_query_fuser = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )

    def forward(self, query, track_query, lane_query, value, query_pos, track_query_pos, lane_query_pos, bbox_results,
                reference_trajs, reference_point_embedding, velocity_embedding, acceleration_embedding, yaw_embedding,
                command_embedding, **kwargs):
        '''
        query: context query (B, A, P, D)
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape

        # Obtain top-k gates and indices from routing network
        top_k_gates, top_k_indices = self.routing_network(query)

        # Prepare tensor to gather selected expert outputs
        all_expert_outputs = torch.zeros((self.num_experts, B * A * P, D), device=query.device)

        # Compute the output of each expert in parallel
        for expert_idx, expert in enumerate(self.experts):
            expert = self.experts[expert_idx]
            if isinstance(expert, TrackInteraction):
                expert_output = expert(query, track_query, query_pos, track_query_pos, pattern="block_sparse")
            elif isinstance(expert, MapInteraction):
                expert_output = expert(query, lane_query, query_pos, lane_query_pos, pattern="block_sparse")
            elif isinstance(expert, VelocityInteraction):
                expert_output = expert(query, velocity_embedding, query_pos, pattern="sliding_window")
            elif isinstance(expert, AccelerationInteraction):
                expert_output = expert(query, acceleration_embedding, query_pos, pattern="sliding_window")
            elif isinstance(expert, YawInteraction):
                expert_output = expert(query, yaw_embedding, query_pos, pattern="sliding_window")
            elif isinstance(expert, ReferencePointInteraction):
                expert_output = expert(query, reference_point_embedding, query_pos, pattern="global_sparse")
            elif isinstance(expert, CommandInteraction):
                expert_output = expert(query, command_embedding, query_pos, pattern="global_sparse")
            else:
                expert_output = expert(query, value, query_pos, bbox_results, reference_trajs, **kwargs)
            all_expert_outputs[expert_idx] = expert_output.reshape(B * A * P, D)

        # Gather and weight top-k expert outputs
        selected_expert_outputs = torch.gather(all_expert_outputs, 0, top_k_indices.unsqueeze(-1).expand(-1, -1, D))
        selected_expert_gates = top_k_gates.unsqueeze(-1)
        weighted_expert_outputs = selected_expert_outputs * selected_expert_gates

        # Sum over experts to get final output
        output = weighted_expert_outputs.sum(dim=1).view(B, A, P, D)

        return self.out_query_fuser(output)

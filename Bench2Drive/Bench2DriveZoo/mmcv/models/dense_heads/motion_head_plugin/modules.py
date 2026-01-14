import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner.base_module import BaseModule
from concurrent.futures import ThreadPoolExecutor
from mmcv.models.utils.functional import (
    norm_points,
    pos2posemb2d,
    features2posemb2d,
    trajectory_coordinate_transform
)
from .InteractionDecoder import InteractionDecoder
import time
from torch.cuda.amp import custom_fwd, custom_bwd

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MotionTransformerDecoder(BaseModule):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """
    def __init__(self, pc_range=None, embed_dims=256, transformerlayers=None, num_layers=3, num_heads=8, num_experts=8, dropout=0.1, batch_first=True, **kwargs):
        super(MotionTransformerDecoder, self).__init__()
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.intention_interaction_layers = IntentionInteraction()

        self.num_experts = num_experts
        self.moa_interaction_layers = nn.ModuleList([
            MoATransformerInteraction(
                num_experts=num_experts,
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=batch_first,
                transformerlayers=transformerlayers,
                ) for _ in range(self.num_layers)
        ])

        self.static_dynamic_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.dynamic_embed_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*3, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.in_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.reference_trajs_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*4, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

        self.build_reference_point_embedding = nn.Sequential(
            nn.Linear(2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.build_velocity_embedding = nn.Sequential(
            nn.Linear(1, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.build_yaw_embedding = nn.Sequential(
            nn.Linear(1, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.build_acceleration_embedding = nn.Sequential(
            nn.Linear(1, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.build_command_embedding = nn.Sequential(
            nn.Linear(1, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
    def forward(self,
                track_query,
                lane_query,
                track_query_pos=None,
                lane_query_pos=None,
                track_bbox_results=None,
                bev_embed=None,
                reference_trajs=None,
                traj_reg_branches=None,
                agent_level_embedding=None,
                scene_level_ego_embedding=None,
                scene_level_offset_embedding=None,
                learnable_embed=None,
                agent_level_embedding_layer=None,
                scene_level_ego_embedding_layer=None,
                scene_level_offset_embedding_layer=None,
                velocity=None, 
                acceleration=None, 
                yaw=None,
                command=None,
                **kwargs):

        """Forward function for `MotionTransformerDecoder`.
        Args:
            agent_query (B, A, D)
            map_query (B, M, D) 
            map_query_pos (B, G, D)
            static_intention_embed (B, A, P, D)
            offset_query_embed (B, A, P, D)
            global_intention_embed (B, A, P, D)
            learnable_intention_embed (B, A, P, D)
            det_query_pos (B, A, D)
            reference_trajs (B, A, P, 12, 2)
        Returns:
            None
        """
        intermediate = []
        intermediate_reference_trajs = []

        # B, _, P, D = agent_level_embedding.shape
        B, A, P, D = agent_level_embedding.shape
        track_query_bc = track_query.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)
        track_query_pos_bc = track_query_pos.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)

        # static intention embedding, which is imutable throughout all layers
        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)
        static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed
        
        # reference_point embedding
        normalized_points = norm_points(reference_trajs, self.pc_range)
        reference_point_embedding = self.build_reference_point_embedding(normalized_points[..., -1, :])
        reference_point_embedding = reference_point_embedding
 
        velocity = velocity[0].float()
        velocity = velocity.view(1, 1, 1).expand(-1, 6, -1)
        velocity_embedding = self.build_velocity_embedding(velocity)

        acceleration = acceleration[0].float()
        acceleration = acceleration[:, 0].unsqueeze(1).unsqueeze(2).expand(-1, 6, -1)
        acceleration_embedding = self.build_acceleration_embedding(acceleration)

        yaw = yaw[0].float()
        yaw = yaw.unsqueeze(1).unsqueeze(2).expand(-1, 6, -1)
        yaw_embedding = self.build_yaw_embedding(yaw)

        command = command[0].float()
        command = command.unsqueeze(1).unsqueeze(2).expand(-1, 6, -1)
        command_embedding = self.build_command_embedding(command)

        reference_trajs_input = reference_trajs.unsqueeze(4).detach()

        query_embed = torch.zeros_like(static_intention_embed) # (B, A, P, D)

        for lid in range(self.num_layers):
            # fuse static and dynamic intention embedding
            # the dynamic intention embedding is the output of the previous layer, which is initialized with anchor embedding
            dynamic_query_embed = self.dynamic_embed_fuser(torch.cat(
                [agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1))
            
            # fuse static and dynamic intention embedding
            query_embed_intention = self.static_dynamic_fuser(torch.cat(
                [static_intention_embed, dynamic_query_embed], dim=-1))  # (B, A, P, D)
            
            # fuse intention embedding with query embedding
            query_embed = self.in_query_fuser(torch.cat([query_embed, query_embed_intention], dim=-1))

            query_embed = self.moa_interaction_layers[lid](
                query = query_embed,
                track_query = track_query,
                lane_query = lane_query,
                query_pos=track_query_pos_bc,   
                track_query_pos = track_query_pos,
                lane_query_pos = lane_query_pos,
                value = bev_embed,  
                bbox_results = track_bbox_results,  
                reference_trajs = reference_trajs_input,
                reference_point_embedding = reference_point_embedding,
                velocity_embedding = velocity_embedding,
                acceleration_embedding = acceleration_embedding,
                yaw_embedding = yaw_embedding,
                command_embedding = command_embedding,
                **kwargs)
            
            if traj_reg_branches is not None:
                # update reference trajectory
                tmp = traj_reg_branches[lid](query_embed)
                bs, n_agent, n_modes, n_steps, _ = reference_trajs.shape
                tmp = tmp.view(bs, n_agent, n_modes, n_steps, -1)
                
                # we predict speed of trajectory and use cumsum trick to get the trajectory
                tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
                new_reference_trajs = torch.zeros_like(reference_trajs)
                new_reference_trajs = tmp[..., :2]
                reference_trajs = new_reference_trajs.detach()
                reference_trajs_input = reference_trajs.unsqueeze(4)  # BS NUM_AGENT NUM_MODE 12 NUM_LEVEL  2

                # update embedding, which is used in the next layer
                # only update the embedding of the last step, i.e. the goal
                ep_offset_embed = reference_trajs.detach()
                ep_ego_embed = trajectory_coordinate_transform(reference_trajs.unsqueeze(
                    2), track_bbox_results, with_translation_transform=True, with_rotation_transform=False).squeeze(2).detach()
                ep_agent_embed = trajectory_coordinate_transform(reference_trajs.unsqueeze(
                    2), track_bbox_results, with_translation_transform=False, with_rotation_transform=True).squeeze(2).detach()

                agent_level_embedding = agent_level_embedding_layer(pos2posemb2d(
                    norm_points(ep_agent_embed[..., -1, :], self.pc_range)))
                scene_level_ego_embedding = scene_level_ego_embedding_layer(pos2posemb2d(
                    norm_points(ep_ego_embed[..., -1, :], self.pc_range)))
                scene_level_offset_embedding = scene_level_offset_embedding_layer(pos2posemb2d(
                    norm_points(ep_offset_embed[..., -1, :], self.pc_range)))

                intermediate.append(query_embed)
                intermediate_reference_trajs.append(reference_trajs)

        return torch.stack(intermediate), torch.stack(intermediate_reference_trajs)

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

        self.embed_dims=embed_dims
        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerEncoderLayer(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
                                                                  batch_first=batch_first)

    def forward(self, query):
        B, A, P, D = query.shape
        # B, A, P, D -> B*A, P, D
        rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = rebatch_x.view(B, A, P, D)
        return out


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
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
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
        query = self.interaction_transformer(query, mem,  pattern)
        query = query.view(B, A, P, D)
        return query

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
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
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
        
        mem = key.mean(dim=1).unsqueeze(1).expand(-1, A * P, -1).contiguous().view(B * A, P, D)

        # B, A, P, D -> B*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem, pattern)
        query = query.view(B, A, P, D)
        return query

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
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
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
        
        mem = key.expand(B*A, -1, -1)
        # B, A, P, D -> B*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem, pattern)
        query = query.view(B, A, P, D)
        return query

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
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
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

        mem = key.expand(B*A, -1, -1)
        # B, A, P, D -> B*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem, pattern)
        query = query.view(B, A, P, D)
        return query

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
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
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
        query = self.interaction_transformer(query, mem, pattern)
        query = query.view(B, A, P, D)
        return query

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
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
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
        query = self.interaction_transformer(query, mem, pattern)
        query = query.view(B, A, P, D)
        return query

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
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.interaction_transformer = InteractionDecoder(d_model=embed_dims,
                                                                  nhead=num_heads,
                                                                  dropout=dropout,
                                                                  dim_feedforward=embed_dims*2,
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

        mem = key.expand(B*A, -1, -1)
        # B, A, P, D -> B*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem, pattern)
        query = query.view(B, A, P, D)
        return query



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
        
        self.embed_dims=embed_dims
        self.batch_first = batch_first
        self.transformerlayers=transformerlayers
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

        query = torch.flatten(query, start_dim=0, end_dim=1)
        
        # Perform cross-attention using query from agents and key, value from BEV embeddings.
        output = self.interaction_transformer(query=query, value=value, query_pos=query_pos, bbox_results=bbox_results, reference_trajs=reference_trajs, **kwargs)
        
        # Reshape output back to (B, A, P, D).
        output = output.view(B, A, P, D)
        
        return output


class RoutingNetwork(nn.Module):
    def __init__(self, embed_dims, num_experts, k, cvloss=0.1, switchloss=0.1, zloss=0.1):
        super().__init__()
        # super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_experts = num_experts
        self.k = k
        self.cvloss = cvloss
        self.switchloss = switchloss
        self.zloss = zloss
        self.w_gate = nn.Parameter(torch.randn(embed_dims, num_experts), requires_grad=True)

    def cv_squared(self, x):
        eps = 1e-10
        return x.float().var() / (x.float().mean()**2 + eps)
    
    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
            F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=0)) ** 2)
        return zloss

    def forward(self, query):
        logits = query @ self.w_gate
        probs = torch.softmax(logits, dim=0)
        top_k_gates, top_k_indices = probs.topk(self.k, dim=0)

        return top_k_gates, top_k_indices

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
                transformerlayers=None,
                k = 4,
                norm_cfg=None,
                init_cfg=None):
        super().__init__()
        
        self.batch_first = batch_first
        self.k = k

        # gating
        self.routing_network = RoutingNetwork(embed_dims, num_experts, k)

        # Parallel_expert
        self.experts = nn.ModuleList([
            TrackInteraction(embed_dims, num_heads, dropout, batch_first),
            MapInteraction(embed_dims, num_heads, dropout, batch_first),
            VelocityInteraction(embed_dims, num_heads, dropout, batch_first),
            AccelerationInteraction(embed_dims, num_heads, dropout, batch_first),
            YawInteraction(embed_dims, num_heads, dropout, batch_first),
            ReferencePointInteraction(embed_dims, num_heads, dropout, batch_first),
            CommandInteraction(embed_dims, num_heads, dropout, batch_first),
            BevInteraction(embed_dims, num_heads, dropout, batch_first,transformerlayers)
        ])
            
        self.num_experts = num_experts
        self.layer_norm = nn.LayerNorm(embed_dims)

        self.out_query_fuser = nn.Sequential(
            nn.Linear(embed_dims*4, embed_dims*4),
            nn.ReLU(),
            nn.Linear(embed_dims*4, embed_dims),
        )


    def forward(self, query, track_query, lane_query, value, query_pos, track_query_pos, lane_query_pos, bbox_results, reference_trajs, reference_point_embedding, velocity_embedding, acceleration_embedding, yaw_embedding, command_embedding, **kwargs):

        B, A, P, D = query.shape
        query_norm = self.layer_norm(query.view(B * A * P, D).mean(dim=0))
   
        top_k_gates, top_k_indices = self.routing_network(query_norm)  
   
        output = []

        # Compute the output of each selected expert 
        for i, expert_idx in enumerate(top_k_indices):  
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
            output.append(expert_output)

        output = torch.cat(output, dim=-1)
        output = self.out_query_fuser(output)

        return output






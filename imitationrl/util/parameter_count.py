import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, MessagePassing
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from vmas.scenarios.navigation import Scenario as BaseNavigation
from torch_geometric.nn import radius_graph
from torch_geometric.nn import GCNConv

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PopArt(nn.Module):
    def __init__(self, input_dim, output_dim, beta=0.99):
        super().__init__()
        self.beta = beta
        self.register_buffer("mean", torch.zeros(output_dim))
        self.register_buffer("mean_sq", torch.zeros(output_dim))
        self.register_buffer("std", torch.ones(output_dim))
        self.v_head = layer_init(nn.Linear(input_dim, output_dim), std=1)

    def forward(self, x):
        return self.v_head(x)

    def update(self, targets):
        # Update statistics and correct weights to preserve unnormalized outputs
        with torch.no_grad():
            batch_mean = targets.mean(dim=0)
            batch_mean_sq = (targets**2).mean(dim=0)
            new_mean = self.beta * self.mean + (1 - self.beta) * batch_mean
            new_mean_sq = self.beta * self.mean_sq + (1 - self.beta) * batch_mean_sq
            new_std = torch.sqrt(torch.clamp(new_mean_sq - new_mean**2, min=1e-5))

            # FIX: Reshape the scale factor to (output_dim, 1) for broadcasting
            scale_factor = (self.std / new_std).view(-1, 1)
            self.v_head.weight.data.mul_(scale_factor)
            
            # Bias is (3,), so this line remains the same
            self.v_head.bias.data.mul_(self.std).add_(self.mean - new_mean).div_(new_std)
            
            self.mean.copy_(new_mean)
            self.mean_sq.copy_(new_mean_sq)
            self.std.copy_(new_std)

    def denormalize(self, x):
        return x * self.std + self.mean
    
    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.std**2 + 1e-8)

class MultiHeadGATBackbone(nn.Module):
    def __init__(self, n_max, feature_dim=8, hidden_dim=64, out_dim=128, heads=4):
        super().__init__()
        self.n_max = n_max
        self.feature_dim = feature_dim
        
        self.gat1 = GATConv(feature_dim, hidden_dim, heads=heads, concat=True, edge_dim=3, add_self_loops=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=3, add_self_loops=True)
        self.gat3 = GATConv(hidden_dim * heads, out_dim, heads=heads, concat=False, edge_dim=3, add_self_loops=True)
        
        self.skip_proj = nn.Linear(feature_dim, hidden_dim * heads)
        self.elu = nn.ELU()

    def _build_fc_dynamic_graph(self, x_flat):
        B = x_flat.shape[0]
        device = x_flat.device
        
        x_padded = x_flat.view(B, self.n_max, self.feature_dim)
        active_mask = x_padded[:, :, 6] > 0.5  
        valid_x = x_padded[active_mask] 
        
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(B, self.n_max)
        valid_batch = batch_indices[active_mask] 
        
        # Memory-safe Block-Diagonal builder (Replaces OOM-prone broadcast)
        local_idx = torch.arange(self.n_max, device=device)
        grid_r, grid_c = torch.meshgrid(local_idx, local_idx, indexing='ij')
        local_edges = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=0)
        
        offsets = (torch.arange(B, device=device) * self.n_max).view(1, B, 1)
        global_edges = (local_edges.unsqueeze(1) + offsets).view(2, -1)
        
        active_flat = active_mask.view(-1)
        valid_edge_mask = active_flat[global_edges[0]] & active_flat[global_edges[1]]
        
        # Strip self-loops explicitly so PyG's add_self_loops can inject synthesized phantom features
        is_not_self = global_edges[0] != global_edges[1]
        valid_edge_mask = valid_edge_mask & is_not_self
        
        padded_edge_index = global_edges[:, valid_edge_mask]
        
        # CRITICAL FIX: Map padded indices back to the compressed valid_x space
        padded_to_active = torch.zeros(B * self.n_max, dtype=torch.long, device=device)
        padded_to_active[active_flat] = torch.arange(valid_x.shape[0], device=device)
        
        edge_index = padded_to_active[padded_edge_index]
        row, col = edge_index
        
        rel_pos = valid_x[row, :2] - valid_x[col, :2]
        distances = torch.sqrt((rel_pos ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        
        # Edge attributes: [dx, dy, distance, is_self_loop]
        edge_attr = torch.cat([rel_pos, distances], dim=-1)
        
        return valid_x, edge_index, edge_attr, valid_batch

    def forward(self, x_flat):
        valid_x, edge_index, edge_attr, valid_batch = self._build_fc_dynamic_graph(x_flat)
        
        res = self.skip_proj(valid_x)
        
        h = self.gat1(valid_x, edge_index, edge_attr=edge_attr)
        h = self.elu(h) + res 
        
        h2 = self.gat2(h, edge_index, edge_attr=edge_attr)
        h2 = self.elu(h2) + h 
        
        node_embeddings = self.gat3(h2, edge_index, edge_attr=edge_attr) 
        
        return valid_x, node_embeddings, valid_batch

class GraphAgent(nn.Module):
    def __init__(self, envs, n_max, num_agents): 
        super().__init__()
        self.num_agents = num_agents
        self.n_max = n_max
        self.action_dim = np.prod((2,))
        
        # GAT
        # self.backbone = MultiHeadGATBackbone(n_max=n_max, feature_dim=9)

        # GCN 
        self.backbone = FairVectorGCNBackbone(n_max=n_max, feature_dim=9)
        gat_out_dim = 128
        
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(gat_out_dim, 128)),
            nn.LayerNorm(128), nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64), nn.ReLU(),
            layer_init(nn.Linear(64,64)),
            nn.ELU(),
        )

        # The Mean Head (Restored Tanh for instant braking)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, self.action_dim), std=0.01),
            nn.Tanh() 
        )

        # The Variance Head
        self.actor_logstd = nn.Parameter(torch.full((1, self.action_dim), -0.5))
        
        # FIX: Critic only sees the invariant graph (128) + density scalar (1)
        critic_in_dim = (gat_out_dim * 3) + 1
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(critic_in_dim, 256)),
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128), nn.ReLU(),
        )
        self.critic_popart = PopArt(128, 1)
        # self.obs_normalizer = GraphObservationNormalizer(n_max=n_max, feature_dim=9, continuous_dim=4)

    def get_value(self, x_flat, denormalize=False):
        return

    def get_action_and_value(self, x_flat, action=None, denormalize=False):
        return

class SeparatedVectorGCNConv(MessagePassing):
    """
    A GraphSAGE-style isotropic convolution that separates the root node from 
    the neighbor average. This preserves 100% of the ego agent's kinematics 
    (velocity/position) while averaging environmental context without attention.
    """
    def __init__(self, in_channels, out_channels, edge_dim=3):
        super().__init__(aggr='mean') 
        # Separate linear projections for the Ego node vs Neighbor nodes
        self.root_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.node_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_proj = nn.Linear(edge_dim, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_attr):
        # 1. Project the ego node independently at 100% strength
        root_out = self.root_proj(x)
        
        # 2. Project neighbors and propagate messages
        x_proj = self.node_proj(x)
        aggr_out = self.propagate(edge_index, x=x_proj, edge_attr=edge_attr)
        
        # 3. Combine Ego Kinematics + Neighbor Context
        return root_out + aggr_out + self.bias

    def message(self, x_j, edge_attr):
        return x_j + self.edge_proj(edge_attr)


class FairVectorGCNBackbone(nn.Module):
    def __init__(self, n_max, feature_dim=8, hidden_dim=64, out_dim=128, heads=4):
        super().__init__()
        self.n_max = n_max
        self.feature_dim = feature_dim
        
        wide_dim = hidden_dim * heads  # 256
        
        # Swap to the Separated Vector GCN
        self.gcn1 = SeparatedVectorGCNConv(feature_dim, wide_dim, edge_dim=3)
        self.gcn2 = SeparatedVectorGCNConv(wide_dim, wide_dim, edge_dim=3)
        self.gcn3 = SeparatedVectorGCNConv(wide_dim, out_dim, edge_dim=3)
        
        self.skip_proj = nn.Linear(feature_dim, wide_dim)
        self.elu = nn.ELU()

    def _build_fc_dynamic_graph(self, x_flat):
        B = x_flat.shape[0]
        device = x_flat.device
        
        x_padded = x_flat.view(B, self.n_max, self.feature_dim)
        active_mask = x_padded[:, :, 6] > 0.5  
        valid_x = x_padded[active_mask] 
        
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(B, self.n_max)
        valid_batch = batch_indices[active_mask] 
        
        local_idx = torch.arange(self.n_max, device=device)
        grid_r, grid_c = torch.meshgrid(local_idx, local_idx, indexing='ij')
        local_edges = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=0)
        
        offsets = (torch.arange(B, device=device) * self.n_max).view(1, B, 1)
        global_edges = (local_edges.unsqueeze(1) + offsets).view(2, -1)
        
        active_flat = active_mask.view(-1)
        valid_edge_mask = active_flat[global_edges[0]] & active_flat[global_edges[1]]
        
        # CRITICAL FIX: Strip self-loops! 
        # Because SeparatedVectorGCNConv handles the root node via self.root_proj, 
        # we MUST strip self-loops from edge_index so the ego node does not get 
        # redundantly mixed into the neighbor averaging pool.
        is_not_self = global_edges[0] != global_edges[1]
        valid_edge_mask = valid_edge_mask & is_not_self
        
        padded_edge_index = global_edges[:, valid_edge_mask]
        
        padded_to_active = torch.zeros(B * self.n_max, dtype=torch.long, device=device)
        padded_to_active[active_flat] = torch.arange(valid_x.shape[0], device=device)
        
        edge_index = padded_to_active[padded_edge_index]
        row, col = edge_index
        
        rel_pos = valid_x[row, :2] - valid_x[col, :2]
        distances = torch.sqrt((rel_pos ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        
        edge_attr = torch.cat([rel_pos, distances], dim=-1)
        
        return valid_x, edge_index, edge_attr, valid_batch

    def forward(self, x_flat):
        valid_x, edge_index, edge_attr, valid_batch = self._build_fc_dynamic_graph(x_flat)
        
        res = self.skip_proj(valid_x)
        
        # Layer 1: Identical skip + ELU routing
        h = self.gcn1(valid_x, edge_index, edge_attr=edge_attr)
        h = self.elu(h) + res 
        
        # Layer 2: Identical residual routing
        h2 = self.gcn2(h, edge_index, edge_attr=edge_attr)
        h2 = self.elu(h2) + h 
        
        # Layer 3: No final residual, matching GAT exactly
        node_embeddings = self.gcn3(h2, edge_index, edge_attr=edge_attr) 
        
        return valid_x, node_embeddings, valid_batch

class TransformerAgent(nn.Module):
    def __init__(self, single_action_space, single_obs_shape, num_agents, state_dim, n_max=10, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = np.array((2*10*9,)).prod() # Exactly 2 * n_max * 9
        self.action_dim = np.prod((2,))
        self.n_max_nodes = n_max 
        self.feature_dim = 9
        self.nhead = nhead # Needed for multi-head distance mask expansion
        
        # 1. SHARED NORMALIZER
        # self.obs_normalizer = GraphObservationNormalizer(n_max=n_max, feature_dim=self.feature_dim, continuous_dim=4)
        
        # 2. UPGRADE: Learnable Spatial Horizon per Attention Head!
        # Initialized to 1.0 so Euclidean distance immediately biases attention on Step 0.
        self.geom_scale = nn.Parameter(torch.ones(1, nhead, 1, 1))
        
        # 3. TERMINAL LAYERNORM TOKEN PROJECTION
        self.token_proj = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim, d_model)),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            layer_init(nn.Linear(d_model, d_model)),
            nn.LayerNorm(d_model) 
        )
        
        # 4. TRANSFORMER ENCODER
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model, 
            batch_first=True, 
            activation='gelu', 
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers, 
            norm=nn.LayerNorm(d_model), 
            enable_nested_tensor=False
        )
        
        # 5. TRI-TOKEN ACTOR HEAD (d_model * 3)
        self.actor = nn.Sequential(
            nn.LayerNorm(d_model * 3), 
            layer_init(nn.Linear(d_model * 3, 128)),
            nn.LayerNorm(128), nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64), nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, self.action_dim), std=0.01),
            nn.Tanh() 
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

        # 6. CENTRALIZED CRITIC
        joint_critic_dim = self.num_agents * self.obs_dim
        self.critic_encoder = nn.Sequential(
            layer_init(nn.Linear(joint_critic_dim, 256)), 
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128), nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64), nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )
        self.critic = PopArt(64, 1)

    def _forward_actor_backbone(self, x_norm):
        B = x_norm.shape[0]
        tokens = x_norm.view(B, self.n_max_nodes, self.feature_dim)
        key_padding_mask = (tokens[:, :, 6] < 0.5)
        
        # --- UPGRADE: Euclidean Attention Biasing ---
        # 1. Extract physical relative coordinates [dx, dy] for all 2K tokens
        pos = tokens[:, :, 0:2] # [B, 2K, 2]
        
        # 2. Calculate Pairwise Euclidean Distance Matrix [B, 1, 2K, 2K]
        pos_i = pos.unsqueeze(2) # [B, 2K, 1, 2]
        pos_j = pos.unsqueeze(1) # [B, 1, 2K, 2]
        dist_matrix = torch.norm(pos_i - pos_j, dim=-1, keepdim=True).permute(0, 3, 1, 2)
        
        # 3. Apply per-head learnable spatial scaling: (-gamma_h * distance) -> [B, nhead, 2K, 2K]
        scaled_dist = -torch.abs(self.geom_scale) * dist_matrix
        
        # 4. Reshape to [B * nhead, 2K, 2K] as required by PyTorch nn.TransformerEncoder
        geom_mask = scaled_dist.view(B * self.nhead, self.n_max_nodes, self.n_max_nodes)
        
        # Pass tokens AND the geometric distance kernel into the Transformer
        h = self.token_proj(tokens)
        out = self.transformer(h, mask=geom_mask, src_key_padding_mask=key_padding_mask)
        
        # Tri-Token Concatenation
        ego_final = out[:, 0, :]
        ego_initial = h[:, 0, :]
        goal_initial = h[:, self.n_max_nodes // 2, :]
        
        combined_ego = torch.cat([ego_final, ego_initial, goal_initial], dim=-1)
        
        return combined_ego

    def get_actor_parameters(self):
        # Includes self.geom_scale so attention heads learn their spatial horizons!
        return (
            [self.geom_scale] +
            list(self.token_proj.parameters()) +
            list(self.transformer.parameters()) +
            list(self.actor.parameters()) +
            list(self.actor_mean.parameters()) +
            [self.actor_logstd]
        )

    def get_value(self, x, denormalize=False):
        return 

    def get_action_and_value(self, x, action=None, denormalize=False):
        return

class MLPAgent(nn.Module):
    def __init__(self, single_action_space, single_obs_shape, num_agents, state_dim, n_max=10):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = np.array((2*10*9,)).prod() # Exactly 2 * n_max * 9
        self.action_dim = np.prod((2,))
        
        # Shared Isotropic Normalizer
        # self.obs_normalizer = GraphObservationNormalizer(n_max=n_max, feature_dim=9, continuous_dim=4)

        # 1. DECENTRALIZED ACTOR (No embeddings, no global state -> 100% invariant to N!)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 256)),
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128), nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64), nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, self.action_dim), std=0.01),
            nn.Tanh() 
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

        # 2. CENTRALIZED CRITIC (Receives joint state of ALL agents to stabilize GAE training!)
        # Notice we use state_dim (or num_agents * obs_dim), which gives it true CTDE clarity.
        joint_critic_dim = self.num_agents * self.obs_dim
        self.critic_encoder = nn.Sequential(
            layer_init(nn.Linear(joint_critic_dim, 256)), 
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128), nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64), nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )
        self.critic = PopArt(64, 1)

    def get_value(self, x, denormalize=False):
        return

    def get_action_and_value(self, x, action=None, denormalize=False):
        return

def count_model_footprint(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    
    print(f"Trainable Params: {trainable}")
    print(f"Non-trainable Params: {non_trainable}")
    print(f"Buffers (PopArt/Normalizer stats): {buffers}")
    
    return trainable + non_trainable + buffers

if __name__ == "__main__":
    agent = GraphAgent(
            envs=1, 
            n_max=10, 
            num_agents=20
        )

    print(count_model_footprint(agent))
    
    agent = MLPAgent(
        single_action_space=1,
        single_obs_shape=1,
        num_agents=20,
        state_dim=1
    )

    print(count_model_footprint(agent))

    agent = TransformerAgent(
        single_action_space=1,
        single_obs_shape=1,
        num_agents=20,
        state_dim=1,
        n_max=10
    )

    print(count_model_footprint(agent))
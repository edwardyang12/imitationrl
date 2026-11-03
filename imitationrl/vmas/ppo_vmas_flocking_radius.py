# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import argparse
import os
import random
import time
import math
import vmas 
import imageio
from distutils.util import strtobool
import gc

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, MessagePassing, radius_graph, GCNConv
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from vmas.scenarios.flocking import Scenario as BaseFlocking

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="vmas-flocking-radius",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="flocking",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=200000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=7e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4096,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--anneal-ent", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.995,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=256,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=5,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--num-landmarks", type=int, default=3,
        help="number of agents and landmarks")
    parser.add_argument("--max-cycles", type=int, default=250,
        help="length of environment run")
    parser.add_argument("--n-max", type=int, default=5, help="Fixed context window size for the GNN")
    parser.add_argument("--comm-radius", type=float, default=0.5, help="Spatial radius for neighbor detection")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

class FlockingCleanObs(BaseFlocking):
    def observation(self, agent):
        # 1. Base kinematics
        obs = [agent.state.pos, agent.state.vel]
        
        # 2. Target Tracking (Direction fixed: Pointing FROM agent TO target)
        obs.append(self._target.state.pos - agent.state.pos)
        obs.append(self._target.state.vel - agent.state.vel)
            
        # 3. Strict Index-Preserved Teammates
        for a in self.world.agents:
            obs.append(a.state.pos - agent.state.pos)
            obs.append(a.state.vel - agent.state.vel)
                
        # 4. Relative Obstacles
        for landmark in self.world.landmarks:
            if landmark != self._target:
                obs.append(landmark.state.pos - agent.state.pos)
                
        return torch.cat(obs, dim=-1)
    
    def reward(self, agent):
        base_reward = super().reward(agent)
        
        # 2. Target Cruising Alignment
        target_match_penalty = -torch.linalg.norm(self._target.state.vel - agent.state.vel, dim=-1)
        
        # 3. Flock Uniform Heading Alignment
        all_vels = torch.stack([a.state.vel for a in self.world.agents], dim=1) 
        mean_vel = all_vels.mean(dim=1) 
        flock_match_penalty = -torch.linalg.norm(mean_vel - agent.state.vel, dim=-1)
        
        # 4. Reynolds Separation / Uniform Spacing
        separation_penalty = 0.0
        desired_spacing = 0.4
        for a in self.world.agents:
            if a != agent:
                dist = torch.linalg.norm(a.state.pos - agent.state.pos, dim=-1)
                intrusion = torch.clamp(desired_spacing - dist, min=0.0)
                separation_penalty -= (intrusion ** 2) * 5.0

        # 5. Reynolds Cohesion
        all_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        com = all_pos.mean(dim=1)
        cohesion_penalty = -torch.linalg.norm(com - agent.state.pos, dim=-1)

        shaped_reward = (
            base_reward 
            + (target_match_penalty * 0.05) 
            + (flock_match_penalty * 0.05) 
            + (separation_penalty * 0.1) 
            + (cohesion_penalty * 0.05)
        )
        return shaped_reward

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GraphObservationNormalizer(nn.Module):
    def __init__(self, n_max, feature_dim=10, continuous_dim=4, epsilon=1e-5):
        super().__init__()
        self.n_max = n_max
        self.feature_dim = feature_dim
        self.continuous_dim = continuous_dim
        
        self.register_buffer("running_var", torch.ones(continuous_dim))
        self.count = epsilon

    def update(self, x_flat):
        B = x_flat.shape[0]
        x_reshaped = x_flat.view(B, self.n_max, self.feature_dim)
        
        active_mask = x_reshaped[:, :, 8] > 0.5 
        
        # 1. SPATIAL ISOTROPIC VARIANCE
        valid_pos = x_reshaped[active_mask][:, :2]
        clamped_pos = torch.clamp(valid_pos, min=-5.0, max=5.0)
        pos_var_scalar = torch.clamp((clamped_pos ** 2).mean(), min=0.1)
        
        # 2. VELOCITY ISOTROPIC VARIANCE (Use all active agent/target nodes with valid velocity)
        vel_mask = ((x_reshaped[:, :, 5] > 0.5) | (x_reshaped[:, :, 6] > 0.5)) & active_mask
        valid_vel = x_reshaped[vel_mask][:, 2:4]
        clamped_vel = torch.clamp(valid_vel, min=-5.0, max=5.0)
        vel_var_scalar = torch.clamp((clamped_vel ** 2).mean(), min=0.1)

        batch_var = torch.stack([pos_var_scalar, pos_var_scalar, vel_var_scalar, vel_var_scalar])
        batch_count = valid_pos.shape[0]

        delta_var = batch_var - self.running_var
        self.running_var = self.running_var + delta_var * (batch_count / (self.count + batch_count))
        self.count += batch_count

    def normalize(self, x_flat):
        B = x_flat.shape[0]
        x_reshaped = x_flat.view(B, self.n_max, self.feature_dim).clone()
        
        active_mask = x_reshaped[:, :, 8] > 0.5
        valid_x = x_reshaped[active_mask] 
        valid_continuous = valid_x[:, :self.continuous_dim]
        
        normalized_continuous = valid_continuous / torch.sqrt(self.running_var + 1e-8)
        
        valid_x[:, :self.continuous_dim] = normalized_continuous
        x_reshaped[active_mask] = valid_x
        
        return x_reshaped.view(B, -1)

class MultiHeadGATBackbone(nn.Module):
    def __init__(self, n_max, feature_dim=10, hidden_dim=64, out_dim=128, heads=4, radius=0.5):
        super().__init__()
        self.n_max = n_max
        self.feature_dim = feature_dim
        self.radius = radius
        
        self.gat1 = GATConv(feature_dim, hidden_dim, heads=heads, concat=True, edge_dim=5, add_self_loops=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=5, add_self_loops=True)
        self.gat3 = GATConv(hidden_dim * heads, out_dim, heads=heads, concat=False, edge_dim=5, add_self_loops=True)
        
        self.skip_proj = nn.Linear(feature_dim, hidden_dim * heads)
        self.elu = nn.ELU()

    def _build_radius_dynamic_graph(self, x_flat):
        B = x_flat.shape[0]
        device = x_flat.device
        
        x_padded = x_flat.view(B, self.n_max, self.feature_dim)
        
        active_mask = x_padded[:, :, 8] > 0.5  
        valid_x = x_padded[active_mask] 
        
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(B, self.n_max)
        valid_batch = batch_indices[active_mask] 
        
        # 1. Create a spatial clone specifically for the radius check
        spatial_x = valid_x[:, :2].clone()
        
        # 2. Index 6 is the 'is_target' flag in Flocking. Teleport target to origin (0,0) so it always connects
        is_target = valid_x[:, 6] > 0.5
        spatial_x[is_target] = 0.0 
        
        # 3. Build edges using the radius graph
        edge_index = radius_graph(
            x=spatial_x, 
            r=self.radius, 
            batch=valid_batch, 
            loop=False
        )
        
        row, col = edge_index
        
        # Network uses UNMODIFIED valid_x for actual edge attributes (pos + dist + vel)
        rel_pos = valid_x[row, :2] - valid_x[col, :2]
        rel_vel = valid_x[row, 2:4] - valid_x[col, 2:4]
        distances = torch.sqrt((rel_pos ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        
        edge_attr = torch.cat([rel_pos, distances, rel_vel], dim=-1)
        
        return valid_x, edge_index, edge_attr, valid_batch

    def forward(self, x_flat):
        valid_x, edge_index, edge_attr, valid_batch = self._build_radius_dynamic_graph(x_flat)
        
        res = self.skip_proj(valid_x)
        
        h = self.gat1(valid_x, edge_index, edge_attr=edge_attr)
        h = self.elu(h) + res 
        
        h2 = self.gat2(h, edge_index, edge_attr=edge_attr)
        h2 = self.elu(h2) + h 
        
        node_embeddings = self.gat3(h2, edge_index, edge_attr=edge_attr) 
        
        return valid_x, node_embeddings, valid_batch

class SeparatedVectorGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=5):
        super().__init__(aggr='mean') 
        self.root_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.node_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_proj = nn.Linear(edge_dim, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_attr):
        root_out = self.root_proj(x)
        x_proj = self.node_proj(x)
        aggr_out = self.propagate(edge_index, x=x_proj, edge_attr=edge_attr)
        return root_out + aggr_out + self.bias

    def message(self, x_j, edge_attr):
        return x_j + self.edge_proj(edge_attr)

class FairVectorGCNBackbone(nn.Module):
    def __init__(self, n_max, feature_dim=10, hidden_dim=64, out_dim=128, heads=4, radius=0.5):
        super().__init__()
        self.n_max = n_max
        self.feature_dim = feature_dim
        self.radius = radius
        
        wide_dim = hidden_dim * heads  # 256
        
        self.gcn1 = SeparatedVectorGCNConv(feature_dim, wide_dim, edge_dim=5)
        self.gcn2 = SeparatedVectorGCNConv(wide_dim, wide_dim, edge_dim=5)
        self.gcn3 = SeparatedVectorGCNConv(wide_dim, out_dim, edge_dim=5)
        
        self.skip_proj = nn.Linear(feature_dim, wide_dim)
        self.elu = nn.ELU()

    def _build_radius_dynamic_graph(self, x_flat):
        B = x_flat.shape[0]
        device = x_flat.device
        
        x_padded = x_flat.view(B, self.n_max, self.feature_dim)
        
        active_mask = x_padded[:, :, 8] > 0.5  
        valid_x = x_padded[active_mask] 
        
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(B, self.n_max)
        valid_batch = batch_indices[active_mask] 
        
        spatial_x = valid_x[:, :2].clone()
        is_target = valid_x[:, 6] > 0.5
        spatial_x[is_target] = 0.0 
        
        edge_index = radius_graph(
            x=spatial_x, 
            r=self.radius, 
            batch=valid_batch, 
            loop=False
        )
        
        row, col = edge_index
        
        rel_pos = valid_x[row, :2] - valid_x[col, :2]
        rel_vel = valid_x[row, 2:4] - valid_x[col, 2:4]
        distances = torch.sqrt((rel_pos ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        
        edge_attr = torch.cat([rel_pos, distances, rel_vel], dim=-1)
        
        return valid_x, edge_index, edge_attr, valid_batch

    def forward(self, x_flat):
        valid_x, edge_index, edge_attr, valid_batch = self._build_radius_dynamic_graph(x_flat)
        
        res = self.skip_proj(valid_x)
        
        h = self.gcn1(valid_x, edge_index, edge_attr=edge_attr)
        h = self.elu(h) + res 
        
        h2 = self.gcn2(h, edge_index, edge_attr=edge_attr)
        h2 = self.elu(h2) + h 
        
        node_embeddings = self.gcn3(h2, edge_index, edge_attr=edge_attr) 
        
        return valid_x, node_embeddings, valid_batch

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
        with torch.no_grad():
            batch_mean = targets.mean(dim=0)
            batch_mean_sq = (targets**2).mean(dim=0)
            new_mean = self.beta * self.mean + (1 - self.beta) * batch_mean
            new_mean_sq = self.beta * self.mean_sq + (1 - self.beta) * batch_mean_sq
            new_std = torch.sqrt(torch.clamp(new_mean_sq - new_mean**2, min=1e-5))

            scale_factor = (self.std / new_std).view(-1, 1)
            self.v_head.weight.data.mul_(scale_factor)
            self.v_head.bias.data.mul_(self.std).add_(self.mean - new_mean).div_(new_std)
            
            self.mean.copy_(new_mean)
            self.mean_sq.copy_(new_mean_sq)
            self.std.copy_(new_std)

    def denormalize(self, x):
        return x * self.std + self.mean
    
    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.std**2 + 1e-8)

class GraphAgent(nn.Module):
    def __init__(self, envs, n_max, num_agents, radius=0.5): 
        super().__init__()
        self.num_agents = num_agents
        self.n_max = n_max
        self.action_dim = np.prod(envs.single_action_space.shape)
        
        # GAT
        self.backbone = MultiHeadGATBackbone(n_max=n_max, feature_dim=10, radius=radius)

        # GCN 
        # self.backbone = FairVectorGCNBackbone(n_max=n_max, feature_dim=10, radius=radius)
        gat_out_dim = 128
        
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(gat_out_dim, 128)),
            nn.LayerNorm(128), nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64), nn.ReLU(),
            layer_init(nn.Linear(64,64)),
            nn.ELU(),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, self.action_dim), std=0.01),
            nn.Tanh() 
        )

        self.actor_logstd = nn.Parameter(torch.full((1, self.action_dim), -0.5))
        
        critic_in_dim = (gat_out_dim * 3) + 1
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(critic_in_dim, 256)),
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128), nn.ReLU(),
        )
        self.critic_popart = PopArt(128, 1)
        self.obs_normalizer = GraphObservationNormalizer(n_max=n_max, feature_dim=10, continuous_dim=4)

    def get_value(self, x_flat, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x_flat)
        valid_x, node_embeddings, valid_batch = self.backbone(x_norm)
        
        batch_size = x_flat.shape[0]
        pooled_mean = global_mean_pool(node_embeddings, valid_batch, size=batch_size)
        pooled_max = global_max_pool(node_embeddings, valid_batch, size=batch_size)

        agent_mask = valid_x[:, 4] > 0.5 
        ego_embeddings = node_embeddings[agent_mask]
        
        active_counts = torch.bincount(valid_batch, minlength=x_flat.shape[0])
        density_scalar_expanded = (active_counts.float() / self.n_max).view(-1, 1)

        critic_input = torch.cat([
            pooled_mean, 
            pooled_max, 
            ego_embeddings, 
            density_scalar_expanded
        ], dim=-1)
        
        values = self.critic_popart(self.critic_mlp(critic_input))
        if denormalize:
            values = self.critic_popart.denormalize(values)
            
        return values.view(-1, 1)

    def get_action_and_value(self, x_flat, action=None, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x_flat)
        valid_x, node_embeddings, valid_batch = self.backbone(x_norm)
        
        agent_mask = valid_x[:, 4] > 0.5
        agent_embeddings = node_embeddings[agent_mask] 
        
        actor_features = self.actor_mlp(agent_embeddings)

        action_means = self.actor_mean(actor_features)
        action_logstds = self.actor_logstd.expand_as(action_means)
        
        safe_logstds = torch.clamp(action_logstds, min=-5.0, max=2.0)
        action_stds = safe_logstds.exp()
        
        probs = Normal(action_means, action_stds)
        
        if action is None:
            action = probs.sample()

        batch_size = x_flat.shape[0]
        pooled_mean = global_mean_pool(node_embeddings, valid_batch, size=batch_size)
        pooled_max = global_max_pool(node_embeddings, valid_batch, size=batch_size)

        active_counts = torch.bincount(valid_batch, minlength=x_flat.shape[0])
        density_scalar_expanded = (active_counts.float() / self.n_max).view(-1, 1)
        
        critic_input = torch.cat([
            pooled_mean, 
            pooled_max, 
            agent_embeddings, 
            density_scalar_expanded
        ], dim=-1)
        
        values = self.critic_popart(self.critic_mlp(critic_input))
        if denormalize:
            values = self.critic_popart.denormalize(values)
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), values.view(-1, 1)

class MAPPOAgent(nn.Module):
    def __init__(self, single_action_space, single_obs_shape, num_agents, state_dim, n_max=5):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = np.array(single_obs_shape).prod()
        self.action_dim = np.prod(single_action_space.shape)
        
        self.obs_normalizer = GraphObservationNormalizer(n_max=n_max, feature_dim=10, continuous_dim=4)

        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 1024)),
            nn.LayerNorm(1024), nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(256, self.action_dim), std=0.01),
            nn.Tanh() 
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

        joint_critic_dim = self.num_agents * self.obs_dim
        self.critic_encoder = nn.Sequential(
            layer_init(nn.Linear(joint_critic_dim, 512)), 
            nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.critic_popart = PopArt(256, 1)

    def get_value(self, x, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x)
        batch_size = x.shape[0]
        num_games = batch_size // self.num_agents
        
        joint_state = x_norm.view(num_games, -1)
        expanded_joint_state = joint_state.repeat_interleave(self.num_agents, dim=0)
        
        values = self.critic_popart(self.critic_encoder(expanded_joint_state)) 
        if denormalize:
            values = self.critic_popart.denormalize(values)
        return values.view(-1, 1)

    def get_action_and_value(self, x, action=None, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x)
        
        actor_features = self.actor(x_norm)
        action_means = self.actor_mean(actor_features)
        action_logstds = self.actor_logstd.expand_as(action_means)
        safe_logstds = torch.clamp(action_logstds, min=-5.0, max=2.0)
        action_stds = safe_logstds.exp()
        
        probs = Normal(action_means, action_stds)
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x, denormalize)

class PointNetAgent(nn.Module):
    def __init__(self, single_action_space, single_obs_shape, num_agents, state_dim, n_max=5):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = np.array(single_obs_shape).prod()
        self.action_dim = np.prod(single_action_space.shape)
        self.n_max_nodes = n_max 
        self.feature_dim = 10
        
        self.obs_normalizer = GraphObservationNormalizer(n_max=n_max, feature_dim=self.feature_dim, continuous_dim=4)
        
        self.phi = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim, 128)),
            nn.LayerNorm(128), nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.LayerNorm(128), nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.LayerNorm(128), nn.ReLU()
        )
        
        self.rho = nn.Sequential(
            layer_init(nn.Linear(128, 128)),
            nn.LayerNorm(128), nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64), nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU()
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, self.action_dim), std=0.01),
            nn.Tanh() 
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

        joint_critic_dim = self.num_agents * self.obs_dim
        self.critic_encoder = nn.Sequential(
            layer_init(nn.Linear(joint_critic_dim, 512)), 
            nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.critic_popart = PopArt(256, 1)

    def _forward_actor_backbone(self, x_norm):
        B = x_norm.shape[0]
        tokens = x_norm.view(B, self.n_max_nodes, self.feature_dim)
        
        is_active = (tokens[:, :, 8] >= 0.5).unsqueeze(-1).float()
        node_features = self.phi(tokens)
        masked_features = node_features + (1.0 - is_active) * -1e9
        global_feature, _ = torch.max(masked_features, dim=1)
        
        return global_feature

    def get_actor_parameters(self):
        return (
            list(self.phi.parameters()) +
            list(self.rho.parameters()) +
            list(self.actor_mean.parameters()) +
            [self.actor_logstd]
        )

    def get_value(self, x, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x)
        batch_size = x.shape[0]
        num_games = batch_size // self.num_agents
        
        joint_state = x_norm.view(num_games, -1)
        expanded_joint_state = joint_state.repeat_interleave(self.num_agents, dim=0)
        
        values = self.critic_popart(self.critic_encoder(expanded_joint_state)) 
        if denormalize:
            values = self.critic_popart.denormalize(values)
        return values.view(-1, 1)

    def get_action_and_value(self, x, action=None, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x)
        global_feature = self._forward_actor_backbone(x_norm)
        actor_features = self.rho(global_feature)
        
        action_means = self.actor_mean(actor_features)
        action_logstds = self.actor_logstd.expand_as(action_means)
        
        safe_logstds = torch.clamp(action_logstds, min=-5.0, max=2.0)
        action_stds = safe_logstds.exp()
        
        probs = Normal(action_means, action_stds)
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x, denormalize)

class TransformerAgent(nn.Module):
    def __init__(self, single_action_space, single_obs_shape, num_agents, state_dim, n_max=5, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = np.array(single_obs_shape).prod()
        self.action_dim = np.prod(single_action_space.shape)
        self.n_max_nodes = n_max 
        self.feature_dim = 10
        self.nhead = nhead
        
        self.obs_normalizer = GraphObservationNormalizer(n_max=n_max, feature_dim=self.feature_dim, continuous_dim=4)
        self.geom_scale = nn.Parameter(torch.ones(1, nhead, 1, 1))
        
        self.token_proj = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim, d_model)),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            layer_init(nn.Linear(d_model, d_model)),
            nn.LayerNorm(d_model) 
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
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

        joint_critic_dim = self.num_agents * self.obs_dim
        self.critic_encoder = nn.Sequential(
            layer_init(nn.Linear(joint_critic_dim, 512)), 
            nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.critic_popart = PopArt(256, 1)

    def _forward_actor_backbone(self, x_norm):
        B = x_norm.shape[0]
        tokens = x_norm.view(B, self.n_max_nodes, self.feature_dim)
        
        key_padding_mask = (tokens[:, :, 8] < 0.5) 
        
        pos = tokens[:, :, 0:2]
        pos_i = pos.unsqueeze(2) 
        pos_j = pos.unsqueeze(1) 
        dist_matrix = torch.norm(pos_i - pos_j, dim=-1, keepdim=True).permute(0, 3, 1, 2)
        scaled_dist = -torch.abs(self.geom_scale) * dist_matrix
        
        float_padding = torch.zeros(B, self.n_max_nodes, device=x_norm.device)
        float_padding[key_padding_mask] = -1e9 
        float_padding = float_padding.unsqueeze(1).unsqueeze(2).expand(B, self.nhead, self.n_max_nodes, self.n_max_nodes).clone()
        
        diag_idx = torch.arange(self.n_max_nodes, device=x_norm.device)
        float_padding[:, :, diag_idx, diag_idx] = 0.0
        
        combined_mask = scaled_dist + float_padding
        geom_mask = combined_mask.view(B * self.nhead, self.n_max_nodes, self.n_max_nodes)
        
        h = self.token_proj(tokens)
        out = self.transformer(h, mask=geom_mask)
        
        # Index 0 is guaranteed Ego (-1e9), Index 1 is guaranteed Target (-1e8)
        ego_final = out[:, 0, :]
        ego_initial = h[:, 0, :]
        target_initial = h[:, 1, :]
        
        combined_ego = torch.cat([ego_final, ego_initial, target_initial], dim=-1)
        return combined_ego

    def get_actor_parameters(self):
        return (
            [self.geom_scale] +
            list(self.token_proj.parameters()) +
            list(self.transformer.parameters()) +
            list(self.actor.parameters()) +
            list(self.actor_mean.parameters()) +
            [self.actor_logstd]
        )

    def get_value(self, x, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x)
        batch_size = x.shape[0]
        num_games = batch_size // self.num_agents
        
        joint_state = x_norm.view(num_games, -1)
        expanded_joint_state = joint_state.repeat_interleave(self.num_agents, dim=0)
        
        values = self.critic_popart(self.critic_encoder(expanded_joint_state)) 
        if denormalize:
            values = self.critic_popart.denormalize(values)
        return values.view(-1, 1)

    def get_action_and_value(self, x, action=None, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x)
        combined_ego = self._forward_actor_backbone(x_norm)
        actor_features = self.actor(combined_ego)
        
        action_means = self.actor_mean(actor_features)
        action_logstds = self.actor_logstd.expand_as(action_means)
        
        safe_logstds = torch.clamp(action_logstds, min=-5.0, max=2.0)
        action_stds = safe_logstds.exp()
        
        probs = Normal(action_means, action_stds)
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x, denormalize)

class VMASVectorizedEnv:
    def __init__(self, args, seed, run_name, update_step=0):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.num_agents = args.num_landmarks
        self.num_games = args.num_envs // self.num_agents
        self.num_envs = self.num_games * self.num_agents
        self.run_name = run_name
        self.update_step = update_step
        
        self.env = vmas.make_env(
            scenario=FlockingCleanObs() if args.env_id == "flocking" else args.env_id,
            num_envs=self.num_games,
            device=self.device,
            continuous_actions=True,
            n_agents=self.num_agents,
            seed=seed,
            dict_spaces=False,
            n_obstacles=0
        )
        
        self.single_action_space = self.env.action_space[0]
        self.n_max = args.n_max
        self.feature_dim = 10
        
        target_dim = self.n_max * self.feature_dim
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(target_dim,))
        
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.step_count = 0
        self.episode_count = 0
        self.record_this_episode = self.args.capture_video
        self.video_frames = []

        self.ep_polarization = torch.zeros(self.num_games, device=self.device)
        self.ep_cohesion = torch.zeros(self.num_games, device=self.device)
        self.ep_tracking = torch.zeros(self.num_games, device=self.device)
        self.ep_collisions = torch.zeros(self.num_games, device=self.device)

    def _apply_graph_formatting(self, stacked_vmas_obs):
        B = stacked_vmas_obs.shape[0]
        N = stacked_vmas_obs.shape[1]
        MAX_CAPACITY = self.n_max 
        radius = self.args.comm_radius
        
        obs_dim = stacked_vmas_obs.shape[-1]
        O = (obs_dim - 8 - 4 * N) // 2 
        
        # 1. Extract Raw Components
        abs_pos = stacked_vmas_obs[:, :, 0:2] 
        vel = stacked_vmas_obs[:, :, 2:4]     
        target_rel_pos = stacked_vmas_obs[:, :, 4:6] 
        target_rel_vel = stacked_vmas_obs[:, :, 6:8] 
        
        # 2. Reconstruct Absolute Positions & Velocities
        target_abs_pos = (abs_pos[:, 0:1, :] + target_rel_pos[:, 0:1, :]) 
        target_abs_vel = (vel[:, 0:1, :] + target_rel_vel[:, 0:1, :]) 
        
        obstacles_rel = stacked_vmas_obs[:, 0:1, 8+4*N:].view(B, O, 2)
        obstacles_abs = abs_pos[:, 0:1, :] + obstacles_rel
        
        all_pos = torch.cat([abs_pos, target_abs_pos, obstacles_abs], dim=1) # (B, N + 1 + O, 2)
        obstacles_vel = torch.zeros((B, O, 2), device=self.device)
        all_vel = torch.cat([vel, target_abs_vel, obstacles_vel], dim=1) 
        
        # 3. Calculate Relative Metrics
        ego_pos = abs_pos.unsqueeze(2) 
        rel_pos = all_pos.unsqueeze(1) - ego_pos 
        distances = torch.norm(rel_pos, dim=-1) 
        
        # 4. THE SURVIVAL OVERRIDE
        batch_idx = torch.arange(B, device=self.device).view(B, 1).expand(B, N)
        ego_idx = torch.arange(N, device=self.device).view(1, N).expand(B, N)
        
        # Force Ego (-1e9) and Target (-1e8) to the top of the capacity window
        distances[batch_idx, ego_idx, ego_idx] = -1e9
        target_idx = torch.full((B, N), N, device=self.device, dtype=torch.long)
        distances[batch_idx, ego_idx, target_idx] = -1e8
        
        # 5. MEMORY PRIORITIZATION
        total_entities = N + 1 + O
        actual_k = min(MAX_CAPACITY, total_entities)
        dist_sorted, topk_idx = torch.topk(distances, k=actual_k, dim=-1, largest=False)
        
        # 6. Build the Dense Feature Matrix (Feature Dim = 10)
        graph = torch.zeros((B, N, total_entities, self.feature_dim), device=self.device)
        graph[..., 0:2] = rel_pos

        ego_vel = vel.unsqueeze(2)
        rel_vel = all_vel.unsqueeze(1) - ego_vel
        graph[..., 2:4] = rel_vel
        
        is_self_matrix = torch.eye(N, device=self.device)
        graph[:, :, :N, 4] = is_self_matrix.unsqueeze(0) # is_self
        graph[:, :, :N, 5] = 1.0 # is_agent
        graph[:, :, N:N+1, 6] = 1.0 # is_target
        if O > 0:
            graph[:, :, N+1:, 7] = 1.0 # is_obstacle
            
        agent_tags = self.episode_tags.unsqueeze(1).expand(B, N, N)
        graph[:, :, :N, 9] = agent_tags
        if total_entities > N:
            graph[:, :, N:, 9] = -1.0
        
        # 7. Gather Top-K prioritized entities
        exp_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.feature_dim)
        gathered_graph = torch.gather(graph, dim=2, index=exp_idx)
        
        # 8. APPLY THE RADIUS MASK
        # Ego (-1e9) and Target (-1e8) automatically pass this check.
        # Teammates and obstacles only pass if within comm_radius.
        gathered_graph[..., 8] = (dist_sorted <= radius).float()
        
        # 9. ZERO-PADDING FIX
        if actual_k < MAX_CAPACITY:
            pad_size = MAX_CAPACITY - actual_k
            padding = torch.zeros((B, N, pad_size, self.feature_dim), device=self.device)
            padding[..., 0:2] = 100.0 # Push ghost nodes 100m away
            padding[..., 8] = 0.0     # Explicitly mark inactive
            gathered_graph = torch.cat([gathered_graph, padding], dim=2)
            
        return gathered_graph.reshape(B * N, -1)

    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
            
        vmas_obs = self.env.reset()
        self.episode_returns.zero_()
        self.step_count = 0

        self.ep_polarization.zero_()
        self.ep_cohesion.zero_()
        self.ep_tracking.zero_()
        self.ep_collisions.zero_()
        
        self.episode_tags = torch.rand((self.num_games, self.num_agents), device=self.device)
        
        stacked_obs = torch.stack(vmas_obs, dim=1)
        final_obs = self._apply_graph_formatting(stacked_obs)
            
        return final_obs, {"raw_obs": stacked_obs.reshape(self.num_envs, -1)}

    def step(self, actions):
        actions_reshaped = actions.view(self.num_games, self.num_agents, -1)
        vmas_actions = [actions_reshaped[:, i, :] for i in range(self.num_agents)]
        
        vmas_obs, vmas_rews, _, vmas_info = self.env.step(vmas_actions)
        self.step_count += 1
        
        rewards = torch.stack(vmas_rews, dim=1).reshape(-1).detach()
        self.episode_returns += rewards
        stacked_obs = torch.stack(vmas_obs, dim=1).detach()

        with torch.no_grad():
            step_metrics = compute_flocking_metrics(stacked_obs)
            self.ep_polarization += step_metrics["polarization"]
            self.ep_cohesion += step_metrics["cohesion_spread"]
            self.ep_tracking += step_metrics["tracking_error"]
            self.ep_collisions += step_metrics["collision_rate"]

        if self.record_this_episode:
            frames = []
            num_to_render = min(9, self.num_games)
            for i in range(num_to_render):
                frame = self.env.render(mode="rgb_array", env_index=i, agent_index_focus=None)
                if isinstance(frame, list): frame = frame[0]
                frames.append(frame)

            n = len(frames)
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            H, W, C = frames[0].shape
            blank = np.zeros((H, W, C), dtype=np.uint8)
            
            while len(frames) < rows * cols:
                frames.append(blank)
                
            grid = np.vstack([np.hstack(frames[i*cols:(i+1)*cols]) for i in range(rows)])
            self.video_frames.append(grid)

        final_obs = self._apply_graph_formatting(stacked_obs)
        
        is_done = self.step_count >= self.args.max_cycles
        
        terminations = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        truncations = torch.full((self.num_envs,), is_done, device=self.device, dtype=torch.bool)
        
        info = {"raw_obs": stacked_obs.reshape(self.num_envs, -1)}
        
        if is_done:
            self.episode_count += 1
            info["terminal_raw_obs"] = info["raw_obs"].clone()
            info["terminal_observation"] = final_obs.clone()

            info["episode_metrics"] = {
                "polarization": (self.ep_polarization / self.step_count).mean().item(),
                "cohesion_spread": (self.ep_cohesion / self.step_count).mean().item(),
                "tracking_error": (self.ep_tracking / self.step_count).mean().item(),
                "collision_rate": (self.ep_collisions / self.step_count).mean().item(),
            }
            
            if self.record_this_episode and self.video_frames:
                os.makedirs(f"videos_vmas_flocking/{self.run_name}", exist_ok=True)
                file_path = f"videos_vmas_flocking/{self.run_name}/rl-video-update_{self.update_step}-ep_{self.episode_count}.mp4"
                imageio.mimsave(file_path, self.video_frames, fps=15)
                self.video_frames = []
            
            if self.args.capture_video and self.episode_count % 50 == 0:
                self.record_this_episode = True
            else:
                self.record_this_episode = False

            info["episode"] = {
                "r": self.episode_returns.clone(),
                "l": torch.full((self.num_envs,), self.step_count, device=self.device)
            }

            vmas_obs = self.env.reset()
            self.episode_returns.zero_()
            self.step_count = 0

            self.ep_polarization.zero_()
            self.ep_cohesion.zero_()
            self.ep_tracking.zero_()
            self.ep_collisions.zero_()
            
            self.episode_tags = torch.rand((self.num_games, self.num_agents), device=self.device)
            stacked_obs = torch.stack(vmas_obs, dim=1)
            final_obs = self._apply_graph_formatting(stacked_obs)
            info["raw_obs"] = stacked_obs.reshape(self.num_envs, -1)
                
        return final_obs, rewards, terminations, truncations, info

    def close(self): pass

def compute_flocking_metrics(stacked_vmas_obs, agent_radius=0.1):
    B = stacked_vmas_obs.shape[0]
    N = stacked_vmas_obs.shape[1]
    
    pos = stacked_vmas_obs[:, :, 0:2]
    vel = stacked_vmas_obs[:, :, 2:4]
    target_rel = stacked_vmas_obs[:, 0:1, 4:6]
    target_abs = pos[:, 0:1, :] + target_rel
    
    speeds = torch.norm(vel, dim=-1, keepdim=True) + 1e-8
    headings = vel / speeds
    mean_headings = headings.sum(dim=1) / N
    polarization = torch.norm(mean_headings, dim=-1)
    
    com = pos.mean(dim=1, keepdim=True)
    cohesion_spread = torch.norm(pos - com, dim=-1).mean(dim=1)
    
    tracking_error = torch.norm(com - target_abs, dim=-1).squeeze(1)
    
    pos_expanded_1 = pos.unsqueeze(2)
    pos_expanded_2 = pos.unsqueeze(1)
    pairwise_distances = torch.norm(pos_expanded_1 - pos_expanded_2, dim=-1)
    
    eye_mask = torch.eye(N, device=pos.device).unsqueeze(0).bool()
    pairwise_distances.masked_fill_(eye_mask, float('inf'))
    
    collision_threshold = agent_radius * 2.0
    is_colliding = (pairwise_distances < collision_threshold).any(dim=-1).float()
    collision_rate = is_colliding.mean(dim=1)
    
    return {
        "polarization": polarization,
        "cohesion_spread": cohesion_spread,
        "tracking_error": tracking_error,
        "collision_rate": collision_rate
    }

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = VMASVectorizedEnv(args, args.seed, run_name, update_step=0)
    print("VMAS Observation Space:", envs.env.observation_space)
    actual_num_envs = envs.num_envs
    num_agents_per_game = envs.num_agents
    num_games = envs.num_games

    args.batch_size = int(actual_num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    num_updates = args.total_timesteps // args.batch_size    

    global_step = 0
    start_time = time.time()
    reset_data = envs.reset(seed=args.seed)
    if isinstance(reset_data, tuple):
        next_obs = reset_data[0].clone().to(device)
        next_info = reset_data[1]
    else:
        next_obs = reset_data.clone().to(device)
        next_info = {}
    next_done = torch.zeros(actual_num_envs).to(device)

    state_dim = num_agents_per_game * np.array(envs.single_observation_space.shape).prod()
    n_max = args.n_max

    # agent = PointNetAgent(
    #     envs.single_action_space, 
    #     envs.single_observation_space.shape, 
    #     num_agents_per_game, 
    #     state_dim=state_dim, 
    #     n_max=n_max
    # ).to(device)

    # agent = MAPPOAgent(
    #     envs.single_action_space, 
    #     envs.single_observation_space.shape, 
    #     num_agents_per_game, 
    #     state_dim=state_dim, 
    #     n_max=n_max
    # ).to(device)

    # agent = TransformerAgent(
    #     envs.single_action_space, 
    #     envs.single_observation_space.shape, 
    #     num_agents_per_game, 
    #     state_dim=state_dim, 
    #     n_max=n_max
    # ).to(device)

    agent = GraphAgent(
        envs=envs, 
        n_max=n_max, 
        num_agents=num_agents_per_game,
        radius=args.comm_radius
    ).to(device)

    optimizer = optim.Adam([
            {'params': list(agent.backbone.parameters()) + 
                       list(agent.actor_mlp.parameters()) + 
                       list(agent.actor_mean.parameters()) + 
                       [agent.actor_logstd], 'lr': 3e-4}, 
            
            {'params': list(agent.critic_mlp.parameters()) + 
                       list(agent.critic_popart.parameters()), 'lr': 1e-3} 
        ], eps=1e-5)

    obs = torch.zeros((args.num_steps, actual_num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, actual_num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    dones = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    values = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    ent_coef_now = 0

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0:
                    initial_lr = 3e-4 
                    param_group["lr"] = max(5e-5, frac * initial_lr)
                else:
                    initial_lr = 1e-3
                    param_group["lr"] = max(1e-4, frac * initial_lr)
                
        if args.anneal_ent:
            progress = (update - 1.0) / num_updates
            if progress < 0.7:
                ent_coef_now = args.ent_coef 
            else:
                decay_progress = (progress - 0.7) / 0.3
                ent_coef_now = max(0.0001, args.ent_coef * (1.0 - decay_progress))
        else:
            ent_coef_now = args.ent_coef

        if update % 100 == 0:
            print(f"--- UPDATE {update}: PERFORMING PHOENIX REBOOT OF ENVIRONMENTS ---")
            envs.close()
            new_seed = args.seed + update 
            envs = VMASVectorizedEnv(args, new_seed, run_name, update_step=update)
            
            reset_data = envs.reset(seed=new_seed)
            if isinstance(reset_data, tuple):
                next_obs = reset_data[0].clone().to(device)
                next_info = reset_data[1]
            else:
                next_obs = reset_data.clone().to(device)
                next_info = {}
            next_done = torch.zeros(actual_num_envs).to(device)
            print("--- PHOENIX REBOOT COMPLETE ---")

        for step in range(0, args.num_steps):
            global_step += actual_num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, denormalize=True)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            clipped_action = torch.clamp(action, -1.0, 1.0)
            step_data = envs.step(clipped_action)
    
            if len(step_data) == 5:
                next_obs, reward, terminations, truncations, next_info = step_data
                done = terminations
                resets = terminations | truncations
            else:
                next_obs, reward, done, next_info = step_data
                resets = done

            rewards[step] = reward.clone().to(device).view(-1)
            next_obs, next_done = next_obs.clone().to(device), done.clone().to(device).float()

            if resets.any() and "episode" in next_info:
                done_bool = resets.bool()
                done_games = done_bool.view(num_games, num_agents_per_game)[:, 0]
                
                if done_games.any():
                    ep_returns = next_info["episode"]["r"].view(num_games, num_agents_per_game)[done_games]
                    ep_lengths = next_info["episode"]["l"].view(num_games, num_agents_per_game)[done_games]
                    
                    avg_return = ep_returns.mean().item()
                    avg_length = ep_lengths.float().mean().item()

                    metrics = next_info["episode_metrics"]
                    writer.add_scalar("charts/polarization", metrics["polarization"], global_step)
                    writer.add_scalar("charts/cohesion_spread", metrics["cohesion_spread"], global_step)
                    writer.add_scalar("charts/tracking_error", metrics["tracking_error"], global_step)
                    writer.add_scalar("charts/collision_rate", metrics["collision_rate"], global_step)
                    
                    print(f"global_step={global_step}, episodic_return={avg_return:.2f}")
                    writer.add_scalar("charts/team_episodic_return", avg_return, global_step)
                    writer.add_scalar("charts/team_episodic_length", avg_length, global_step)

        with torch.no_grad():
            boot_obs = next_obs.clone()
            if "terminal_observation" in next_info:
                boot_obs = next_info["terminal_observation"].clone()

            next_value = agent.get_value(boot_obs, denormalize=True).flatten()
    
            temp_advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                temp_advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            
            b_returns = (temp_advantages + values).reshape(-1)
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)

            agent.critic_popart.update(b_returns.view(-1, 1))
            agent.obs_normalizer.update(b_obs)

            b_obs_flat = obs.view(-1, obs.shape[-1])
            new_values_flat = torch.zeros(b_obs_flat.shape[0], device=device)
            
            chunk_size = args.batch_size // args.num_minibatches 
            for start in range(0, b_obs_flat.shape[0], chunk_size):
                end = start + chunk_size
                new_values_flat[start:end] = agent.get_value(b_obs_flat[start:end], denormalize=True).flatten()
                
            new_values = new_values_flat.view(args.num_steps, actual_num_envs)
            new_next_value = agent.get_value(boot_obs, denormalize=True).flatten()
            
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = new_next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = new_values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - new_values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + new_values
            values = new_values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        clipfracs = []
        num_joint_steps = args.batch_size // agent.num_agents
        joint_inds = np.arange(num_joint_steps)

        for epoch in range(args.update_epochs):
            rng.shuffle(joint_inds)
            for start in range(0, num_joint_steps, args.minibatch_size // agent.num_agents):
                end = start + (args.minibatch_size // agent.num_agents)
                mb_joint_inds = joint_inds[start:end]
                mb_inds = (mb_joint_inds[:, None] * agent.num_agents + np.arange(agent.num_agents)).flatten()

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv_reshaped = mb_advantages.view(-1, agent.num_agents)
                    mb_adv_reshaped = (mb_adv_reshaped - mb_adv_reshaped.mean(dim=0)) / (mb_adv_reshaped.std(dim=0) + 1e-7)
                    mb_advantages = mb_adv_reshaped.reshape(-1)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                normalized_returns = agent.critic_popart.normalize(b_returns[mb_inds].view(-1, 1)).reshape(-1)
                normalized_values = agent.critic_popart.normalize(b_values[mb_inds].view(-1, 1)).reshape(-1)

                v_loss_unclipped = (newvalue - normalized_returns) ** 2
                if args.clip_vloss:
                    v_clipped = normalized_values + torch.clamp(
                        newvalue - normalized_values, -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - normalized_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * v_loss_unclipped.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        if update % 500 == 0:
            gc.collect()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("charts/ent_coef_now", ent_coef_now, global_step)

        if update % 75 == 0 or update == num_updates:
            save_dir = f"models/{run_name}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(agent.state_dict(), f"{save_dir}/{update}_model.pth")

    envs.close()
    writer.close()

    print("--- STARTING LONG INFERENCE EVALUATION ---")
    agent.eval() 
    
    eval_max_cycles = 1500
    eval_num_agents = 50
    
    eval_env = vmas.make_env(
        scenario=FlockingCleanObs() if args.env_id == "flocking" else args.env_id,
        num_envs=1,
        device=device,
        continuous_actions=True,
        n_agents=eval_num_agents,
        seed=args.seed + 100,
        dict_spaces=False,
        n_obstacles=0
    )
    
    obs_list = eval_env.reset()
    stacked_obs = torch.stack(obs_list, dim=1)
    
    eval_tags = torch.rand((1, eval_num_agents), device=device)
    envs.episode_tags = eval_tags
    next_obs = envs._apply_graph_formatting(stacked_obs)
    
    frames = []
    with torch.no_grad():
        for step in range(eval_max_cycles):
            action, _, _, _ = agent.get_action_and_value(next_obs)
            clipped_action = torch.clamp(action, -1.0, 1.0)
            
            actions_reshaped = clipped_action.view(1, eval_num_agents, -1)
            vmas_actions = [actions_reshaped[:, i, :] for i in range(eval_num_agents)]
            
            vmas_obs, _, _, _ = eval_env.step(vmas_actions)
            
            frame = eval_env.render(mode="rgb_array", env_index=0)
            if isinstance(frame, list): 
                frame = frame[0]
            frames.append(frame)
            
            stacked_obs = torch.stack(vmas_obs, dim=1)
            next_obs = envs._apply_graph_formatting(stacked_obs)

    os.makedirs(f"videos_vmas_flocking/{run_name}", exist_ok=True)
    video_path = f"videos_vmas_flocking/{run_name}/FINAL_LONG_EVAL_{eval_max_cycles}_cycles.mp4"
    imageio.mimsave(video_path, frames, fps=15)
    print(f"--- LONG EVALUATION SAVED TO {video_path} ---")
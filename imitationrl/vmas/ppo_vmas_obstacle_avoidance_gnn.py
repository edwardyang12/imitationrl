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
from torch_geometric.nn import GATConv, global_mean_pool, GATv2Conv, global_max_pool
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from vmas.scenarios.navigation import Scenario as BaseNavigation
from torch_geometric.nn import radius_graph
from torch_geometric.nn import GCNConv
from vmas.simulator.core import Sphere, Landmark
from vmas.scenarios.navigation import Scenario as BaseNavigation

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
    parser.add_argument("--wandb-project-name", type=str, default="vmas",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="navigation",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=300000000,
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
    parser.add_argument("--gamma", type=float, default=0.99,
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
    parser.add_argument("--reward-cheat", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will have extra reward cheats")
    parser.add_argument("--n-max", type=int, default=5, help="Fixed context window size for the GNN")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

class ObstacleAvoidanceScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.random = kwargs.get("random", False)
        self.n_agents = kwargs.get("n_agents", 5) # Default to your args.num_landmarks
        self.n_obstacles = 1
        self.desired_distance = 0.15
        self.min_collision_distance_reward = 1.0

        world = World(batch_dim, device)

        # 1. Single Shared Goal
        self.goal = Landmark(name="goal", collide=False, color=Color.BLACK)
        world.add_landmark(self.goal)

        # 2. Obstacles
        self.obstacles = []
        for i in range(self.n_obstacles):
            obstacle = Landmark(name=f"obstacle_{i}", collide=True, color=Color.RED)
            self.obstacles.append(obstacle)
            world.add_landmark(obstacle)
        
        # 3. Agents
        for i in range(self.n_agents):
            agent = Agent(name=f"agent{i}", collide=True, color=Color.GREEN)
            agent.goal = self.goal # Assign the shared goal
            world.add_agent(agent)

        return world
    
    def generate_grid(self, center: torch.Tensor, num_points: int, distance: float):
        x_center, y_center = center[0].item(), center[1].item()
        num_cols = math.ceil(math.sqrt(num_points))
        num_rows = math.ceil(num_points / num_cols)

        grid = []
        for i in range(num_rows):
            for j in range(num_cols):
                x = x_center + (j - (num_cols - 1) / 2) * distance
                y = y_center + (i - (num_rows - 1) / 2) * distance
                grid.append([x, y])
                if len(grid) >= num_points:
                    break
            if len(grid) >= num_points:
                break
        
        # Ensure the generated grid is on the correct GPU device
        return torch.tensor(grid, device=center.device)

    def reset_world_at(self, env_index: int = None):
        # Match their exact hardcoded positions
        self.goal.set_pos(torch.tensor([-0.8, 0.8], device=self.world.device), batch_index=env_index)
        self.obstacles[0].set_pos(torch.tensor([-0.1, 0.1], device=self.world.device), batch_index=env_index)

        delta = torch.normal(mean=0.0, std=0.1, size=(2,), device=self.world.device) if self.random else torch.tensor([0.0, 0.0], device=self.world.device)
        central_position = torch.tensor([0.6, -0.6], device=self.world.device) + delta

        all_agents_positions = self.generate_grid(central_position, self.n_agents, self.desired_distance)

        for i, agent in enumerate(self.world.agents):
            agent.set_pos(all_agents_positions[i], batch_index=env_index)

    def reward(self, agent: Agent):
        return self.distance_to_goal_reward(agent) + 2.5 * self.obstacle_avoidance_reward(agent)
    
    def distance_to_goal_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        return -agent.distance_to_goal

    def obstacle_avoidance_reward(self, agent: Agent):
        obs_rew = torch.zeros(self.world.batch_dim, device=self.world.device)
        # Vectorized equivalent of their target logic
        for i in range(1, self.n_obstacles + 1):
            obstacle = self.world.landmarks[i]
            dist = self.world.get_distance(agent, obstacle)
            
            # Apply penalty only to environments where distance <= 1.0
            penalty_mask = dist <= self.min_collision_distance_reward
            obs_rew[penalty_mask] -= (self.min_collision_distance_reward - dist[penalty_mask])
            
        return obs_rew

    def observation(self, agent: Agent):
        # Outputs standard relative vectors so your GNN formatting continues to work
        obs = [agent.state.pos, agent.state.vel, agent.state.pos - agent.goal.state.pos]
        
        for a in self.world.agents:
            if a != agent:
                obs.append(a.state.pos - agent.state.pos)
                
        for o in self.obstacles:
            obs.append(o.state.pos - agent.state.pos)
            
        return torch.cat(obs, dim=-1)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GraphObservationNormalizer(nn.Module):
    def __init__(self, n_max, feature_dim=7, continuous_dim=4, epsilon=1e-5):
        super().__init__()
        self.n_max = n_max
        self.feature_dim = feature_dim
        self.continuous_dim = continuous_dim
        
        # Track ONLY variance. Relative coordinates must always remain centered at 0.0
        shape = (continuous_dim,)
        self.register_buffer("running_var", torch.ones(shape))
        self.count = epsilon

    def update(self, x_flat):
        B = x_flat.shape[0]
        x_reshaped = x_flat.view(B, self.n_max, self.feature_dim)
        
        active_mask = x_reshaped[:, :, 6] > 0.5 
        valid_x = x_reshaped[active_mask] 
        continuous_x = valid_x[:, :self.continuous_dim] 
        
        # RMS Update: Variance around zero
        batch_var = (continuous_x ** 2).mean(dim=0)
        batch_count = continuous_x.shape[0]

        delta_var = batch_var - self.running_var
        self.running_var = self.running_var + delta_var * (batch_count / (self.count + batch_count))
        self.count += batch_count

    def normalize(self, x_flat):
        B = x_flat.shape[0]
        x_reshaped = x_flat.view(B, self.n_max, self.feature_dim).clone()
        
        active_mask = x_reshaped[:, :, 6] > 0.5
        valid_x = x_reshaped[active_mask] 
        valid_continuous = valid_x[:, :self.continuous_dim]
        
        # FIX: Only scale by std. Do NOT subtract a mean. Ego stays exactly at 0.0!
        normalized_continuous = valid_continuous / torch.sqrt(self.running_var + 1e-8)
        
        valid_x[:, :self.continuous_dim] = normalized_continuous
        x_reshaped[active_mask] = valid_x
        
        return x_reshaped.view(B, -1)

class MultiHeadGATBackbone(nn.Module):
    def __init__(self, n_max, feature_dim=8, hidden_dim=64, out_dim=128, heads=4):
        super().__init__()
        self.n_max = n_max
        self.feature_dim = feature_dim
        
        # Edge dim is now 3: [delta_x, delta_y, distance]
        self.gat1 = GATConv(feature_dim, hidden_dim, heads=heads, concat=True, edge_dim=3)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=3)
        self.gat3 = GATConv(hidden_dim * heads, out_dim, heads=heads, concat=False, edge_dim=3)
        
        self.skip_proj = nn.Linear(feature_dim, hidden_dim * heads)
        self.elu = nn.ELU()

    def _build_fc_dynamic_graph(self, x_flat):
        B = x_flat.shape[0]
        device = x_flat.device
        
        x_padded = x_flat.view(B, self.n_max, self.feature_dim)
        active_mask = x_padded[:, :, 6] > 0.5  # Assuming index 6 is the active flag
        valid_x = x_padded[active_mask] 
        
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(B, self.n_max)
        valid_batch = batch_indices[active_mask] 
        
        # 1. Create Fully Connected Edge Index per environment
        # Broadcast comparison to find nodes belonging to the same batch
        is_same_batch = valid_batch.unsqueeze(1) == valid_batch.unsqueeze(0)
        
        # Optional: Remove self-loops (GAT usually handles them, but good for clean relative math)
        is_same_batch.fill_diagonal_(False)
        edge_index = is_same_batch.nonzero(as_tuple=False).t().contiguous()
        
        # 2. Extract Relative Edge Attributes [delta_x, delta_y, distance]
        row, col = edge_index
        
        # Assuming valid_x[:, :2] holds the actual positions during graph construction
        # NOTE: If valid_x[:, :2] is already a relative vector to the ego, 
        # you must reconstruct the absolute positions temporarily just to build the graph, 
        # or structure your VMAS wrapper to pass raw positions in just for edge calculation.
        
        rel_pos = valid_x[row, :2] - valid_x[col, :2]
        distances = torch.sqrt((rel_pos ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        
        # Edge attributes: [dx, dy, distance]
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

class SimpleGCNBackbone(nn.Module):
    def __init__(self, n_max, feature_dim=9, hidden_dim=64, out_dim=128):
        super().__init__()
        self.n_max = n_max
        self.feature_dim = feature_dim
        
        # GCNConv accepts 1D edge_weight instead of multidimensional edge_attr
        self.gcn1 = GCNConv(feature_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, out_dim)
        
        self.skip_proj = nn.Linear(feature_dim, hidden_dim)
        self.elu = nn.ELU()

    def _build_fc_dynamic_graph(self, x_flat):
        B = x_flat.shape[0]
        device = x_flat.device
        
        x_padded = x_flat.view(B, self.n_max, self.feature_dim)
        active_mask = x_padded[:, :, 6] > 0.5  # Index 6 is the active flag
        valid_x = x_padded[active_mask] 
        
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(B, self.n_max)
        valid_batch = batch_indices[active_mask] 
        
        # 1. Create Fully Connected Edge Index per environment
        is_same_batch = valid_batch.unsqueeze(1) == valid_batch.unsqueeze(0)
        is_same_batch.fill_diagonal_(False)
        edge_index = is_same_batch.nonzero(as_tuple=False).t().contiguous()
        
        # 2. Extract Relative Distances for GCN Edge Weights
        row, col = edge_index
        rel_pos = valid_x[row, :2] - valid_x[col, :2]
        distances = torch.sqrt((rel_pos ** 2).sum(dim=-1) + 1e-8)
        
        # Use inverse distance as a heuristic edge weight. 
        # This creates a strong spatial baseline without learned attention.
        edge_weight = 1.0 / distances
        
        return valid_x, edge_index, edge_weight, valid_batch

    def forward(self, x_flat):
        valid_x, edge_index, edge_weight, valid_batch = self._build_fc_dynamic_graph(x_flat)
        
        res = self.skip_proj(valid_x)
        
        h = self.gcn1(valid_x, edge_index, edge_weight=edge_weight)
        h = self.elu(h) + res 
        
        h2 = self.gcn2(h, edge_index, edge_weight=edge_weight)
        h2 = self.elu(h2) + h 
        
        node_embeddings = self.gcn3(h2, edge_index, edge_weight=edge_weight) 
        
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

class GraphAgent(nn.Module):
    def __init__(self, envs, n_max, num_agents): 
        super().__init__()
        self.num_agents = num_agents
        self.n_max = n_max
        self.action_dim = np.prod(envs.single_action_space.shape)
        
        # GAT
        # self.backbone = MultiHeadGATBackbone(n_max=n_max, feature_dim=9)

        # GCN 
        self.backbone = SimpleGCNBackbone(n_max=n_max, feature_dim=9)
        gat_out_dim = 128
        
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(gat_out_dim, 128)),
            nn.LayerNorm(128), nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64), nn.ReLU(),
            layer_init(nn.Linear(64,64)),
            nn.ReLU(),
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
        self.obs_normalizer = GraphObservationNormalizer(n_max=n_max, feature_dim=9, continuous_dim=4)

    def get_value(self, x_flat, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x_flat)
        valid_x, node_embeddings, valid_batch = self.backbone(x_norm)
        
        # 1. Global Context (Poolings)
        pooled_mean = global_mean_pool(node_embeddings, valid_batch)
        pooled_max = global_max_pool(node_embeddings, valid_batch)

        # 2. Local Ego Context
        agent_mask = valid_x[:, 4] > 0.5 # Assuming index 4 is 'is_self' or 'is_agent'
        ego_embeddings = node_embeddings[agent_mask]
        
        # 3. Network Density
        active_counts = torch.bincount(valid_batch, minlength=x_flat.shape[0])
        density_scalar_expanded = (active_counts.float() / self.n_max).view(-1, 1)

        # 4. Concatenate for Ego-Aware God-View
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
        
        # Loosely bound the logstd to prevent NaN explosions
        safe_logstds = torch.clamp(action_logstds, min=-5.0, max=2.0)
        action_stds = safe_logstds.exp()
        
        probs = Normal(action_means, action_stds)
        
        if action is None:
            action = probs.sample()
            
        pooled_mean = global_mean_pool(node_embeddings, valid_batch)
        pooled_max = global_max_pool(node_embeddings, valid_batch)

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
            scenario=ObstacleAvoidanceScenario() if args.env_id == "avoidance" else args.env_id,
            num_envs=self.num_games,
            device=self.device,
            continuous_actions=True,
            n_agents=args.num_landmarks,
            collisions=True,
            observe_all_goals=False,
            seed=seed,
            dict_spaces=False,
            world_spawning_x=2,
            world_spawning_y=2
        )
        
        self.single_action_space = self.env.action_space[0]
        self.n_max = args.n_max
        self.feature_dim = 9
        
        # K Agents + 1 Goal + K Obstacles = (K * 2) + 1
        num_graph_nodes = (self.n_max * 2) + 1 
        target_dim = num_graph_nodes * self.feature_dim 
        
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(target_dim,))
        
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.step_count = 0
        self.episode_count = 0
        self.record_this_episode = self.args.capture_video
        self.video_frames = []

    def _apply_graph_formatting(self, stacked_vmas_obs):
        B = self.num_games
        N = self.num_agents
        K = self.args.n_max 
        
        # 1. Dynamically determine the number of obstacles
        # Raw obs = 2(pos) + 2(vel) + 2(goal) + (N-1)*2(agents) + O*2(obstacles)
        # Total = 4 + 2N + 2O
        obs_dim = stacked_vmas_obs.shape[-1]
        num_obstacles = (obs_dim - (2 * N) - 4) // 2
        
        # 2. Extract specific segments
        abs_pos = stacked_vmas_obs[:, :, 0:2] 
        vel = stacked_vmas_obs[:, :, 2:4]     
        raw_ego_to_goal = stacked_vmas_obs[:, :, 4:6] 
        
        # 3. Agents: Relative Positions & KNN
        ego_pos = abs_pos.unsqueeze(2) 
        rel_agents = abs_pos.unsqueeze(1) - ego_pos 
        dist_agents = torch.norm(rel_agents, dim=-1) 
        
        # Ensure ego is always index 0 by setting distance to -infinity
        batch_idx = torch.arange(B, device=self.device).view(B, 1).expand(B, N)
        ego_idx = torch.arange(N, device=self.device).view(1, N).expand(B, N)
        dist_agents[batch_idx, ego_idx, ego_idx] = -1e9
        
        actual_k_agents = min(K, N)
        _, topk_agents_idx = torch.topk(dist_agents, k=actual_k_agents, dim=-1, largest=False) 
        
        # 4. Obstacles: Relative Positions & KNN
        if num_obstacles > 0:
            rel_obstacles = stacked_vmas_obs[:, :, -2 * num_obstacles:].view(B, N, num_obstacles, 2)
            dist_obstacles = torch.norm(rel_obstacles, dim=-1)
            actual_k_obs = min(K, num_obstacles)
            _, topk_obstacles_idx = torch.topk(dist_obstacles, k=actual_k_obs, dim=-1, largest=False)
        else:
            actual_k_obs = 0
            rel_obstacles = torch.zeros((B, N, 0, 2), device=self.device)
            topk_obstacles_idx = torch.zeros((B, N, 0), dtype=torch.long, device=self.device)

        # 5. Build Matrices
        # --- Agents Matrix ---
        graph_agents = torch.zeros((B, N, N, self.feature_dim), device=self.device)
        graph_agents[..., 0:2] = rel_agents
        graph_agents[..., 4] = torch.eye(N, device=self.device).view(1, N, N) # is_self flag
        graph_agents[..., 6] = 1.0 # active
        graph_agents[..., 7] = self.episode_tags.unsqueeze(1).expand(B, N, N)
        
        exp_agents_idx = topk_agents_idx.unsqueeze(-1).expand(-1, -1, -1, self.feature_dim)
        gathered_agents = torch.gather(graph_agents, dim=2, index=exp_agents_idx)
        gathered_agents[:, :, 0, 2:4] = vel # Inject velocity specifically into Ego node
        
        # Pad if there are fewer agents than the context window size
        if actual_k_agents < K:
            padding = torch.zeros((B, N, K - actual_k_agents, self.feature_dim), device=self.device)
            gathered_agents = torch.cat([gathered_agents, padding], dim=2)

        # --- Goal Matrix (Exactly 1 Node) ---
        # Note: raw_ego_to_goal from VMAS is (pos - goal_pos). 
        # We invert it to (goal_pos - pos) so the network sees "Vector FROM Ego TO Target".
        gathered_goal = torch.zeros((B, N, 1, self.feature_dim), device=self.device)
        gathered_goal[..., 0:2] = -raw_ego_to_goal.unsqueeze(2)
        gathered_goal[..., 5] = 1.0 # is_goal flag
        gathered_goal[..., 6] = 1.0 # active
        gathered_goal[..., 7] = self.episode_tags.unsqueeze(1).expand(B, N, 1)
        
        # --- Obstacles Matrix ---
        graph_obstacles = torch.zeros((B, N, num_obstacles, self.feature_dim), device=self.device)
        graph_obstacles[..., 0:2] = rel_obstacles
        graph_obstacles[..., 5] = -1.0 # is_obstacle flag (using -1.0 to distinguish from goals)
        graph_obstacles[..., 6] = 1.0  
        graph_obstacles[..., 7] = self.episode_tags.unsqueeze(1).expand(B, N, num_obstacles)

        if num_obstacles > 0:
            exp_obs_idx = topk_obstacles_idx.unsqueeze(-1).expand(-1, -1, -1, self.feature_dim)
            gathered_obstacles = torch.gather(graph_obstacles, dim=2, index=exp_obs_idx)
        else:
            gathered_obstacles = torch.zeros((B, N, 0, self.feature_dim), device=self.device)
            
        if actual_k_obs < K:
            obs_padding = torch.zeros((B, N, K - actual_k_obs, self.feature_dim), device=self.device)
            gathered_obstacles = torch.cat([gathered_obstacles, obs_padding], dim=2)
            
        # 6. Combine into final graph
        # Total Nodes = K (Agents) + 1 (Goal) + K (Obstacles)
        graph = torch.cat([gathered_agents, gathered_goal, gathered_obstacles], dim=2) 
            
        return graph.reshape(self.num_envs, -1)

    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
            
        vmas_obs = self.env.reset()
        self.episode_returns.zero_()
        self.step_count = 0
        
        # Generate new random tags for tracking identity across the episode
        self.episode_tags = torch.rand((self.num_games, self.num_agents), device=self.device)
        
        stacked_obs = torch.stack(vmas_obs, dim=1)
        final_obs = self._apply_graph_formatting(stacked_obs)
            
        return final_obs, {"raw_obs": stacked_obs.reshape(self.num_envs, -1)}

    def step(self, actions):
        actions_reshaped = actions.view(self.num_games, self.num_agents, -1)
        vmas_actions = [actions_reshaped[:, i, :] for i in range(self.num_agents)]
        
        vmas_obs, vmas_rews, _, vmas_info = self.env.step(vmas_actions)
        self.step_count += 1
        
        rewards = torch.stack(vmas_rews, dim=1).reshape(-1)
        self.episode_returns += rewards

        # --- RESTORED VIDEO LOGIC ---
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
        # ----------------------------

        stacked_obs = torch.stack(vmas_obs, dim=1)
        final_obs = self._apply_graph_formatting(stacked_obs)
        
        is_done = self.step_count >= self.args.max_cycles
        
        # Navigation has no terminal states, only time truncations
        terminations = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        truncations = torch.full((self.num_envs,), is_done, device=self.device, dtype=torch.bool)
        
        info = {"raw_obs": stacked_obs.reshape(self.num_envs, -1)}
        
        if is_done:
            self.episode_count += 1
            info["terminal_raw_obs"] = info["raw_obs"].clone()
            info["terminal_observation"] = final_obs.clone()
            
            # --- RESTORED VIDEO SAVE LOGIC ---
            if self.record_this_episode and self.video_frames:
                os.makedirs(f"videos/{self.run_name}", exist_ok=True)
                file_path = f"videos/{self.run_name}/rl-video-update_{self.update_step}-ep_{self.episode_count}.mp4"
                imageio.mimsave(file_path, self.video_frames, fps=15)
                self.video_frames = []
            
            if self.args.capture_video and self.episode_count % 50 == 0:
                self.record_this_episode = True
            else:
                self.record_this_episode = False
            # ---------------------------------

            info["episode"] = {
                "r": self.episode_returns.clone(),
                "l": torch.full((self.num_envs,), self.step_count, device=self.device)
            }

            vmas_obs = self.env.reset()
            self.episode_returns.zero_()
            self.step_count = 0
            
            self.episode_tags = torch.rand((self.num_games, self.num_agents), device=self.device)
            stacked_obs = torch.stack(vmas_obs, dim=1)
            final_obs = self._apply_graph_formatting(stacked_obs)
            info["raw_obs"] = stacked_obs.reshape(self.num_envs, -1)
                
        return final_obs, rewards, terminations, truncations, info

    def close(self): pass

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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    rng = np.random.default_rng(args.seed)
    current_ratio = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = VMASVectorizedEnv(args, args.seed, run_name, update_step=0)
    print("VMAS Observation Space:", envs.env.observation_space)
    actual_num_envs = envs.num_envs
    num_agents_per_game = envs.num_agents
    num_games = envs.num_games

    args.batch_size = int(actual_num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    num_updates = args.total_timesteps // args.batch_size    

    # TRY NOT TO MODIFY: start the game
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

    if "global_state" in next_info:
        # Use the actual shape of the global state provided by your wrappers
        # state_dim = next_info["global_state"].shape[-1]
        state_dim = num_agents_per_game * np.array(envs.single_observation_space.shape).prod()

        state0 = next_info["global_state"][0]
        state1 = next_info["global_state"][num_agents_per_game] if len(next_info["global_state"]) > 1 else None
        if state1 is not None:
            diff = torch.abs(state0 - state1).sum().item()
            print(f"DEBUG: Environmental Divergence Score: {diff}")
            if diff == 0:
                print("WARNING: Environments are still synchronized!")
    else:
        # Fallback for Atari or environments without a God-view state
        state_dim = num_agents_per_game * np.array(envs.single_observation_space.shape).prod()

    n_max = args.n_max * 2

    agent = GraphAgent(
        envs=envs, 
        n_max=n_max, 
        num_agents=num_agents_per_game
    ).to(device)

    optimizer = optim.Adam([
            {'params': list(agent.backbone.parameters()) + 
                       list(agent.actor_mlp.parameters()) + 
                       list(agent.actor_mean.parameters()) + 
                       [agent.actor_logstd], 'lr': 3e-4}, 
            
            {'params': list(agent.critic_mlp.parameters()) + 
                       list(agent.critic_popart.parameters()), 'lr': 1e-3} 
        ], eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, actual_num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, actual_num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    dones = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    values = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    ent_coef_now = 0

    for update in range(1, num_updates + 1):
        current_assignments = torch.arange(args.num_landmarks).unsqueeze(0).repeat(num_games, 1).to(device)
        needs_assignment = torch.ones(num_games, dtype=torch.bool).to(device)
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for i, param_group in enumerate(optimizer.param_groups):
                # We fetch the initial_lr we set in the Adam constructor
                # If we didn't store it, we can use the current group's base
                if i == 0:
                    initial_lr = 3e-4 
                    param_group["lr"] = max(5e-5, frac * initial_lr)
                else:
                    initial_lr = 1e-3
                    param_group["lr"] = max(1e-4, frac * initial_lr)
                
        if args.anneal_ent:
            progress = (update - 1.0) / num_updates
            
            # HOLD entropy steady while the environment gets harder
            if progress < 0.7:
                ent_coef_now = args.ent_coef 
            # DECAY rapidly only after the curriculum is finished to harden the policy
            else:
                decay_progress = (progress - 0.7) / 0.3
                ent_coef_now = max(0.0001, args.ent_coef * (1.0 - decay_progress))
        else:
            ent_coef_now = args.ent_coef

        if update % 100 == 0:
            print(f"--- UPDATE {update}: PERFORMING PHOENIX REBOOT OF ENVIRONMENTS ---")
            envs.close()
            
            progress = (update - 1.0) / num_updates
            if progress < 0.7:
                # Starts at 0.1, grows to 0.5
                current_ratio = 0.1 + (0.4 * (progress / 0.7))
            else:
                current_ratio = 0.5

            # Rebuild with a staggered seed so we don't repeat the exact same scenarios
            new_seed = args.seed + update 
            envs = VMASVectorizedEnv(args, new_seed, run_name, update_step=update)
            
            # Re-initialize the starting observations for PPO
            reset_data = envs.reset(seed=new_seed)
            if isinstance(reset_data, tuple):
                next_obs = reset_data[0].clone().to(device)
                next_info = reset_data[1]
            else:
                next_obs = reset_data.clone().to(device)
                next_info = {}
            next_done = torch.zeros(actual_num_envs).to(device)
            
            # Re-sync the global state tracking
            if "global_state" in next_info:
                current_game_states = next_obs.view(num_games, -1)
            print("--- PHOENIX REBOOT COMPLETE ---")

        for step in range(0, args.num_steps):
            global_step += actual_num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, denormalize=True)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            clipped_action = torch.clamp(action, -1.0, 1.0)

            # TRY NOT TO MODIFY: execute the game and log data.
            step_data = envs.step(clipped_action)
    
            # Gymnasium step returns (obs, reward, terminations, truncations, infos)
            if len(step_data) == 5:
                next_obs, reward, terminations, truncations, next_info = step_data
                done = terminations

                resets = terminations | truncations
            else:
                next_obs, reward, done, next_info = step_data
                resets = done

            if args.reward_cheat:
                agent_radius = 0.15
                collision_penalty = 0.5
                
                cheat_raw_obs = next_info["raw_obs"].clone()
                if "final_info" in next_info:
                    for i, fin_info in enumerate(next_info["final_info"]):
                        if fin_info is not None and "raw_obs" in fin_info:
                            cheat_raw_obs[i] = fin_info["raw_obs"].clone()

                raw_obs_tensor = torch.Tensor(cheat_raw_obs).to(device)
                new_landmark_dist = raw_obs_tensor.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
                new_landmark_dist = new_landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
                
                # Shape: [num_games, num_agents, num_landmarks]
                dist = torch.norm(new_landmark_dist, dim=-1)
                
                # --- 1. EPISODIC STATIC ASSIGNMENT ---
                # Check which games just reset (or are at step 0)
                game_resets = resets.view(num_games, num_agents_per_game)[:, 0]
                needs_assignment = needs_assignment | game_resets
                
                # If any game reset, recalculate its optimal shortest-path routing
                if needs_assignment.any():
                    dist_clone = dist.clone()
                    
                    # Only calculate for games that need it
                    for g in range(num_games):
                        if needs_assignment[g]:
                            for _ in range(args.num_landmarks):
                                flat_dist = dist_clone[g].view(-1)
                                min_val, min_idx = flat_dist.min(dim=0)
                                
                                agent_idx = min_idx // args.num_landmarks
                                landmark_idx = min_idx % args.num_landmarks
                                
                                current_assignments[g, agent_idx] = landmark_idx
                                
                                dist_clone[g, agent_idx, :] = float('inf')
                                dist_clone[g, :, landmark_idx] = float('inf')
                                
                    needs_assignment.fill_(False)
                
                # Extract the distance to the frozen, optimally assigned target
                assigned_dist = torch.gather(dist, 2, current_assignments.unsqueeze(-1)).squeeze(-1)
                
                # --- 2. PROCEDURAL SQUARE ROOT CALCULUS ---
                # Calculate the mathematical tipping point for the dead-center snap
                min_snap_multiplier = collision_penalty / np.sqrt(agent_radius)
                
                # Apply a safety factor (> 1.0) to guarantee the snap overpowers engine physics
                snap_factor = 1.5 
                procedural_multiplier = min_snap_multiplier * snap_factor
                
                # R = -M * sqrt(d)
                individual_pull = -procedural_multiplier * torch.sqrt(assigned_dist + 1e-8)
                
                # Combine with native environment reward
                rewards[step] = torch.tensor(reward).to(device).view(-1) + individual_pull.view(-1)
            else:
                # normalized_step_rewards = reward_norm(reward, done)
                rewards[step] = reward.clone().to(device).view(-1)
            next_obs, next_done = next_obs.clone().to(device), done.clone().to(device).float()

            # LOGGING TEAM DATA
            if resets.any() and "episode" in next_info:
                done_bool = resets.bool()
                # Grab the first agent's index for each completed game to avoid duplicate logs
                done_games = done_bool.view(num_games, num_agents_per_game)[:, 0]
                
                if done_games.any():
                    avg_return = next_info["episode"]["r"].view(num_games, num_agents_per_game)[done_games, 0].mean().item()
                    avg_length = next_info["episode"]["l"].view(num_games, num_agents_per_game)[done_games, 0].float().mean().item()
                    
                    print(f"global_step={global_step}, episodic_return={avg_return}")
                    writer.add_scalar("charts/team_episodic_return", avg_return, global_step)
                    writer.add_scalar("charts/team_episodic_length", avg_length, global_step)
        # bootstrap value if not done
        with torch.no_grad():
            boot_obs = next_obs.clone()
            
            # TELEPORTATION FIX: If the environment timed out, we MUST bootstrap 
            # from the true final state (step 250), not the reset state (step 0).
            if "terminal_observation" in next_info:
                boot_obs = next_info["terminal_observation"].clone()

            # First Pass
            next_value = agent.get_value(boot_obs, denormalize=True).flatten()
    
            # Standard GAE to get returns
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
            
            # This is our optimization target
            b_returns = (temp_advantages + values).reshape(-1)
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)

            # 2. UPDATE STATS NOW (Before SGD)1
            # This aligns the normalizers with the data we just collected
            agent.critic_popart.update(b_returns.view(-1, 1))

            agent.obs_normalizer.update(b_obs)

            # 3. Second Pass: RE-CALCULATE Values and Advantages with NEW stats
            # This is the crucial step you were missing. 
            # It ensures 'values' and 'returns' are in the same normalized space for SGD.
            # new_values = agent.get_value(
            #     obs.view(-1, obs.shape[-1]),  # <- Use obs.shape[-1] instead of agent.obs_dim
            #     denormalize=True
            # ).view(args.num_steps, actual_num_envs)

            b_obs_flat = obs.view(-1, obs.shape[-1])
            new_values_flat = torch.zeros(b_obs_flat.shape[0], device=device)
            
            # Process the massive batch in safe chunks to protect GAT memory
            chunk_size = args.batch_size // args.num_minibatches 
            for start in range(0, b_obs_flat.shape[0], chunk_size):
                end = start + chunk_size
                new_values_flat[start:end] = agent.get_value(b_obs_flat[start:end], denormalize=True).flatten()
                
            new_values = new_values_flat.view(args.num_steps, actual_num_envs)
            new_next_value = agent.get_value(boot_obs, denormalize=True).flatten()
            
            # Final GAE calculation for the actual SGD update
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
            values = new_values # Use the re-calculated values for the SGD 'b_values'

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        # 1. Determine the number of 'joint-steps' (all agents in a game at one time)
        num_joint_steps = args.batch_size // agent.num_agents
        joint_inds = np.arange(num_joint_steps)

        for epoch in range(args.update_epochs):
            rng.shuffle(joint_inds)
            for start in range(0, num_joint_steps, args.minibatch_size // agent.num_agents):
                end = start + (args.minibatch_size // agent.num_agents)
                # Pick joint indices and expand them to include all agents in those games
                mb_joint_inds = joint_inds[start:end]
                
                # This ensures we always pick Agent 0, 1, 2... from the same game/time together
                mb_inds = (mb_joint_inds[:, None] * agent.num_agents + np.arange(agent.num_agents)).flatten()
                mb_state_inds = mb_joint_inds

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv_reshaped = mb_advantages.view(-1, agent.num_agents)
                    mb_adv_reshaped = (mb_adv_reshaped - mb_adv_reshaped.mean(dim=0)) / (mb_adv_reshaped.std(dim=0) + 1e-7)
                    mb_advantages = mb_adv_reshaped.reshape(-1)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)

                # Reshape to align with PopArt agent-specific stats
                normalized_returns = agent.critic_popart.normalize(b_returns[mb_inds].view(-1, 1)).reshape(-1)
                normalized_values = agent.critic_popart.normalize(b_values[mb_inds].view(-1, 1)).reshape(-1)

                # Standard individual value loss
                v_loss_unclipped = (newvalue - normalized_returns) ** 2
                
                # NEW: Value Decomposition Loss
                # Reshape to [Minibatch_Games, num_agents]
                # nv_reshaped = newvalue.view(-1, agent.num_agents)
                # nr_reshaped = normalized_returns.view(-1, agent.num_agents)
                
                # Penalize the difference between Sum(Predicted Values) and Sum(Actual Returns)
                # joint_v_loss = 0.5 * ((nv_reshaped.sum(dim=1) - nr_reshaped.sum(dim=1)) ** 2).mean()

                if args.clip_vloss:
                    v_clipped = normalized_values + torch.clamp(
                        newvalue - normalized_values, -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - normalized_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * v_loss_unclipped.mean()

                # Combine with a weight for the joint loss
                # total_v_loss = v_loss + 0.01 * joint_v_loss 
                
                entropy_loss = entropy.mean()
                # loss = pg_loss - ent_coef_now * entropy_loss + total_v_loss * args.vf_coef
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

        # TRY NOT TO MODIFY: record rewards for plotting purposes
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

        if update % 100 == 0 or update == num_updates:
            save_dir = f"models/{run_name}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(agent.state_dict(), f"{save_dir}/{update}_model.pth")

    envs.close()
    writer.close()

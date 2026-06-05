# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import argparse
import importlib
import os
import random
import time
from distutils.util import strtobool
import gc

import gymnasium as gym
import numpy as np
import cv2
import supersuit as ss
from supersuit.vector import ConcatVecEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv, global_mean_pool
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.utils.wrappers import BaseParallelWrapper

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
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL_gnn",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="pong_v3",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=70000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=7e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--anneal-ent", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.03,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--num-landmarks", type=int, default=3,
        help="number of agents and landmarks")
    parser.add_argument("--max-cycles", type=int, default=70,
        help="length of environment run")
    parser.add_argument("--reward-cheat", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will have extra reward cheats")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


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
        
        self.gat1 = GATConv(feature_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.gat3 = GATConv(hidden_dim * heads, out_dim, heads=heads, concat=False)
        
        # ADD: Linear projection to match dimensions for the first skip connection
        self.skip_proj = nn.Linear(feature_dim, hidden_dim * heads)
        self.elu = nn.ELU()

    def _build_dynamic_graph(self, x_flat):
        """
        Converts the flat CleanRL tensor into a dynamic, dummy-free PyG Graph.
        Optimized to prevent O(V^2) VRAM explosions during massive Minibatch updates.
        """
        B = x_flat.shape[0]
        device = x_flat.device
        
        # Reshape to [Batch, N_max, Feature_Dim]
        x_padded = x_flat.view(B, self.n_max, self.feature_dim)
        
        # 1. Mask out dummy nodes
        active_mask = x_padded[:, :, 6] > 0.5
        
        # 2. Extract strictly valid node features
        valid_x = x_padded[active_mask] # Shape: [Total_Valid_Nodes_in_Batch, 7]
        
        # 3. Create PyG Batch Indexing
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(B, self.n_max)
        valid_batch = batch_indices[active_mask] 
        
        # 4. Generate Fully Connected Edge Index (Optimized Tile Method)
        # Instead of a dense 46k x 46k matrix, we build one small graph and mathematically tile it B times.
        num_active = int(active_mask[0].sum().item())
        
        # Base graph for a single environment (e.g., 6x6)
        base_edge_index = torch.ones(num_active, num_active, device=device).nonzero(as_tuple=False).t().contiguous()
        
        # Add the node offset for each batch item
        batch_offsets = (torch.arange(B, device=device) * num_active).view(1, B, 1)
        edge_index = base_edge_index.unsqueeze(1).repeat(1, B, 1) + batch_offsets
        
        # Flatten into the standard PyG [2, E] format
        edge_index = edge_index.view(2, -1)
        
        return valid_x, edge_index, valid_batch

    def forward(self, x_flat):
        valid_x, edge_index, valid_batch = self._build_dynamic_graph(x_flat)
        
        # Project original features for the skip connection
        res = self.skip_proj(valid_x)
        
        # Layer 1 + Skip
        h = self.gat1(valid_x, edge_index)
        h = self.elu(h) + res 
        
        # Layer 2 + Skip (Dimensions already match here, so we just add the previous 'h')
        h2 = self.gat2(h, edge_index)
        h2 = self.elu(h2) + h 
        
        # Layer 3 (Final Output)
        node_embeddings = self.gat3(h2, edge_index) 
        
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

class ObservationNormalizer(nn.Module):
    def __init__(self, shape, epsilon=1e-5):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        # Update running mean and variance using Welford's algorithm
        delta = batch_mean - self.running_mean
        new_mean = self.running_mean + delta * batch_count / (self.count + batch_count)
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)
        
        self.running_mean = new_mean
        self.running_var = m_2 / (self.count + batch_count)
        self.count += batch_count

    def normalize(self, x):
        return (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)

class GraphObservationWrapper(BaseParallelWrapper):
    def __init__(self, env, n_max, current_n):
        super().__init__(env)
        self.n_max = n_max
        self.current_n = current_n
        self.feature_dim = 8  # FIX: Bumped from 7 to 8 to hold the Agent ID

        self.random_tags = []
        
        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.n_max * self.feature_dim,), dtype=np.float32
            ) for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def process_obs(self, raw_obs, ego_idx):
        N = self.current_n
        nodes = np.zeros((self.n_max, self.feature_dim), dtype=np.float32)
        
        self_vel = raw_obs[0:2]
        
        # 1. Fetch the Ego Agent's specific Random Tag
        ego_tag = self.random_tags[ego_idx]
        
        # [rel_x, rel_y, vel_x, vel_y, is_self, is_landmark, is_active, RANDOM_TAG]
        nodes[0] = [0.0, 0.0, self_vel[0], self_vel[1], 1.0, 0.0, 1.0, ego_tag]
        
        idx = 1
        lm_start = 4
        lm_end = 4 + 2 * N
        lm_rel_pos = raw_obs[lm_start:lm_end].reshape(N, 2)
        for i in range(N):
            nodes[idx] = [lm_rel_pos[i][0], lm_rel_pos[i][1], 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
            idx += 1
            
        oa_pos_start = lm_end
        oa_pos_end = oa_pos_start + 2 * (N - 1)
        oa_rel_pos = raw_obs[oa_pos_start:oa_pos_end].reshape(N - 1, 2)
        
        oa_vel_start = oa_pos_end
        oa_vel_end = oa_vel_start + 2 * (N - 1)
        oa_vel = raw_obs[oa_vel_start:oa_vel_end].reshape(N - 1, 2)
        
        # 2. Extract the other agents' Tags using MPE's internal ordering (skipping ego)
        other_tags = [self.random_tags[j] for j in range(N) if j != ego_idx]
        
        for i in range(N - 1):
            nodes[idx] = [oa_rel_pos[i][0], oa_rel_pos[i][1], oa_vel[i][0], oa_vel[i][1], 0.0, 0.0, 1.0, other_tags[i]]
            idx += 1
            
        return nodes.flatten()

    def step(self, actions):
        obs, rews, terms, truncs, infos = self.env.step(actions)
        for agent in self.agents:
            infos[agent]["raw_obs"] = obs[agent] 
            
        graph_obs = {}
        for i, agent in enumerate(self.agents):
            graph_obs[agent] = self.process_obs(obs[agent], ego_idx=i)
            
        return graph_obs, rews, terms, truncs, infos

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        
        # FIX: Generate N random scalar tags for the entire episode
        # This keeps the IDs consistent across all decentralized observations
        self.random_tags = np.random.rand(self.current_n)
        
        for agent in self.agents:
            infos[agent]["raw_obs"] = obs[agent]
            
        graph_obs = {}
        for i, agent in enumerate(self.agents):
            # Pass the ego index so the agent knows which tag belongs to it
            graph_obs[agent] = self.process_obs(obs[agent], ego_idx=i)
            
        return graph_obs, infos

class GraphAgent(nn.Module):
    def __init__(self, envs, n_max, num_agents, occupancy_dim): # Remove state_dim
        super().__init__()
        self.num_agents = num_agents
        self.n_max = n_max
        self.occupancy_dim = occupancy_dim
        
        self.backbone = MultiHeadGATBackbone(n_max=n_max, feature_dim=8)
        gat_out_dim = 128
        
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(gat_out_dim, 128)),
            nn.LayerNorm(128), nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64), nn.ReLU(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        )
        
        # FIX: Critic only sees the invariant graph + occupancy flags
        critic_in_dim = gat_out_dim + occupancy_dim 
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(critic_in_dim, 256)),
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128), nn.ReLU(),
        )
        self.critic_popart = PopArt(128, self.num_agents)
        self.obs_normalizer = GraphObservationNormalizer(n_max=n_max, feature_dim=8, continuous_dim=4)

    def get_value(self, x_flat, centralized_state, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x_flat)
        _, node_embeddings, valid_batch = self.backbone(x_norm)
        
        pooled_graph = global_mean_pool(node_embeddings, valid_batch)
        pooled_graph = pooled_graph.view(-1, self.num_agents, pooled_graph.shape[-1]).mean(dim=1)

        # FIX: Simply extract the occupancy flags. No continuous state normalization needed.
        occupancy_flags = centralized_state[:, -self.occupancy_dim:]
        centralized_state_vector = torch.cat([pooled_graph, occupancy_flags], dim=-1)
        
        values = self.critic_popart(self.critic_mlp(centralized_state_vector))
        if denormalize:
            values = self.critic_popart.denormalize(values)
        return values.view(-1, 1)

    def get_action_and_value(self, x_flat, centralized_state, action=None, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x_flat)
        valid_x, node_embeddings, valid_batch = self.backbone(x_norm)
        
        agent_mask = valid_x[:, 4] > 0.5
        agent_embeddings = node_embeddings[agent_mask] 
        
        logits = self.actor_mlp(agent_embeddings)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        pooled_graph = global_mean_pool(node_embeddings, valid_batch)
        pooled_graph = pooled_graph.view(-1, self.num_agents, pooled_graph.shape[-1]).mean(dim=1)

        # FIX: Simply extract the occupancy flags.
        occupancy_flags = centralized_state[:, -self.occupancy_dim:]
        centralized_state_vector = torch.cat([pooled_graph, occupancy_flags], dim=-1)
        
        values = self.critic_popart(self.critic_mlp(centralized_state_vector))
        if denormalize:
            values = self.critic_popart.denormalize(values)
            
        return action, probs.log_prob(action), probs.entropy(), values.view(-1, 1)

    def load_bc_weights(self, bc_model_path="student_bc_best_gnn.pt"):
        # Custom loading logic since the StudentActor only has the backbone + actor_mlp
        bc_state_dict = torch.load(bc_model_path)
        agent_dict = self.state_dict()
        
        pretrained_dict = {k: v for k, v in bc_state_dict.items() if k in agent_dict}
        agent_dict.update(pretrained_dict)
        self.load_state_dict(agent_dict)
        print(f"\n[ORACLE] Successfully injected GNN BC weights!")

class StateWrapper(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Calculate and store state_space so it persists after vectorization
        self.state_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=self.env.unwrapped.state().shape
        )

    def step(self, actions):
        obs, rews, terms, truncs, infos = self.env.step(actions)
        state = self.env.unwrapped.state()
        for agent in self.agents:
            infos[agent]["global_state"] = state
        return obs, rews, terms, truncs, infos

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        state = self.env.unwrapped.state()
        for agent in self.agents:
            infos[agent]["global_state"] = state
        return obs, infos

# 1. Base Environment Factory
def make_env(args, env_id, seed, current_local_ratio=0.5):
    def thunk():
        if "simple_spread" in env_id:
            from mpe2 import simple_spread_v3
            env = simple_spread_v3.parallel_env(N=args.num_landmarks, max_cycles=args.max_cycles, 
                                                local_ratio=current_local_ratio, dynamic_rescaling=True, render_mode="rgb_array")
            env = StateWrapper(env)

            max_nodes = args.num_landmarks * 2 
            env = GraphObservationWrapper(env, n_max=max_nodes, current_n=args.num_landmarks)
        else:
            env = importlib.import_module(f"pettingzoo.atari.{env_id}").parallel_env(render_mode="rgb_array")
            env = ss.max_observation_v0(env, 2)
            env = ss.frame_skip_v0(env, 4)
            env = ss.color_reduction_v0(env, mode="B")
            env = ss.resize_v1(env, x_size=84, y_size=84)
            env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
            env = ss.frame_stack_v1(env, 4)
        
        # SuperSuit natively converts (Agents, Obs) -> (Batch, Obs)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        return env
    return thunk

# 4. SURGICAL WRAPPER: Inherits from VectorEnv to safely bypass Gymnasium type-checks
class DictInfoWrapper(gym.vector.VectorEnv):
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.render_mode = "rgb_array"
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 5}
        self._is_vector_env = True
        self.target_video_size = None

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if isinstance(info, list):
            global_state_list = [i.get("global_state") for i in info]
            raw_obs_list = [i.get("raw_obs") for i in info]
            
            # FIX: Safely extract true final observations and infos
            final_obs_list = [i.get("final_observation", None) for i in info]
            final_info_list = [i.get("final_info", None) for i in info]
            
            new_info = {
                "global_state": np.array(global_state_list), 
                "raw_obs": np.array(raw_obs_list),
                "agents": info
            }
            
            # Only add them if an episode actually ended to prevent KeyErrors
            if any(x is not None for x in final_obs_list):
                new_info["final_observation"] = final_obs_list
            if any(x is not None for x in final_info_list):
                new_info["final_info"] = final_info_list
                
            info = new_info
        return obs, rew, term, trunc, info
        
    def reset(self, seed=None, options=None):
        # If a seed is provided, we must manually stagger it for the sub-environments
        # Because ConcatVecEnv usually passes the same seed to all workers
        if seed is not None:
            # We access the internal list of environments in ConcatVecEnv
            # And call reset on each with a unique staggered seed
            obs_list = []
            info_list = []
            for i, sub_env in enumerate(self.env.vec_envs):
                o, f = sub_env.reset(seed=seed + i*1000, options=options)
                obs_list.append(o)
                info_list.append(f)
            
            # Combine observations and infos manually to match VectorEnv format
            obs = np.concatenate(obs_list)
            info = [item for sublist in info_list for item in (sublist if isinstance(sublist, list) else [sublist])]
        else:
            # If no seed, fall back to default reset
            obs, info = self.env.reset(options=options)
            
        if isinstance(info, list):
            global_state_list = [i.get("global_state") for i in info]
            raw_obs_list = [i.get("raw_obs") for i in info]
            info = {
                "global_state": np.array(global_state_list), 
                "raw_obs": np.array(raw_obs_list),
                "agents": info
            }
        return obs, info
        
    def render(self):
        # 1. Safely attempt to get raw frames
        try:
            game_frames = [sub_env.render() for sub_env in self.env.vec_envs]
        except Exception:
            game_frames = [None for _ in self.env.vec_envs]
        
        processed_frames = []
        for f in game_frames:
            valid_frame = False
            
            # FIX: Hardcode a safe maximum resolution. 
            # 256x256 in a 6x6 grid = 1536x1536 video (Well below the 4096px H.264 limit)
            if self.target_video_size is None:
                self.target_video_size = (256, 256)

            if f is not None:
                try:
                    img = np.array(f, dtype=np.uint8, copy=True)
                    
                    # Handle dimensionality safely
                    if len(img.shape) >= 2:
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        elif len(img.shape) >= 3:
                            img = img[:, :, :3]
                            if img.shape[2] == 1:
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            elif img.shape[2] == 2:
                                pad = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
                                img = np.concatenate([img, pad], axis=-1)
                        
                        # Guaranteed resize down to safe resolution
                        if (img.shape[1], img.shape[0]) != self.target_video_size:
                            img = cv2.resize(img, self.target_video_size, interpolation=cv2.INTER_LINEAR)
                        
                        # Force contiguous memory and explicitly check the byte count
                        img = np.ascontiguousarray(img[:, :, :3], dtype=np.uint8)
                        expected_bytes = self.target_video_size[0] * self.target_video_size[1] * 3
                        
                        if len(img.tobytes()) == expected_bytes:
                            processed_frames.append(img)
                            valid_frame = True
                except Exception:
                    pass 
            
            # Impervious Fallback
            if not valid_frame:
                blank = np.zeros((self.target_video_size[1], self.target_video_size[0], 3), dtype=np.uint8)
                processed_frames.append(blank)
                
        num_agents_per_game = self.num_envs // len(self.env.vec_envs)
        return [f for f in processed_frames for _ in range(num_agents_per_game)]
        
    def close(self): 
        return self.env.close()

def build_environments(args, run_name, base_seed, current_local_ratio=0.5, update_step=0):
    temp_env = make_env(args, args.env_id, base_seed, current_local_ratio)()
    num_agents_per_game = temp_env.num_envs
    num_games = args.num_envs // num_agents_per_game

    env_list = [make_env(args, args.env_id, base_seed + i, current_local_ratio) for i in range(num_games)]
    envs = ConcatVecEnv(env_list)
    envs = DictInfoWrapper(envs)

    envs.is_vector_env = True
    
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    if args.capture_video:
        envs = gym.wrappers.vector.RecordVideo(
            envs, 
            f"videos_gnn/{run_name}", 
            episode_trigger=lambda x: x % 2000 == 0,
            name_prefix=f"rl-video-update_{update_step}"
        )

    # Set Final CleanRL attributes
    envs.single_action_space = temp_env.action_space
    envs.single_observation_space = temp_env.observation_space
    temp_env.close()
    
    envs.is_vector_env = True
    return envs, num_agents_per_game, num_games

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

    strict_occupancy_radius = 0.2

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    rng = np.random.default_rng(args.seed)
    current_ratio = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs, num_agents_per_game, num_games = build_environments(args, run_name, args.seed, current_ratio, update_step=0)
    actual_num_envs = envs.num_envs

    args.batch_size = int(actual_num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    num_updates = args.total_timesteps // args.batch_size    

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    reset_data = envs.reset(seed=args.seed)
    if isinstance(reset_data, tuple):
        next_obs = torch.Tensor(reset_data[0]).to(device)
        next_info = reset_data[1]
    else:
        next_obs = torch.Tensor(reset_data).to(device)
        next_info = {}
    next_done = torch.zeros(actual_num_envs).to(device)

    if "global_state" in next_info:
        # Use the actual shape of the global state provided by your wrappers
        # state_dim = next_info["global_state"].shape[-1]
        state_dim = next_info["global_state"].shape[-1] + args.num_landmarks

        state0 = next_info["global_state"][0]
        state1 = next_info["global_state"][num_agents_per_game] if len(next_info["global_state"]) > 1 else None
        if state1 is not None:
            diff = np.abs(state0 - state1).sum()
            print(f"DEBUG: Environmental Divergence Score: {diff}")
            if diff == 0:
                print("WARNING: Environments are still synchronized!")
    else:
        # Fallback for Atari or environments without a God-view state
        state_dim = num_agents_per_game * np.array(envs.single_observation_space.shape).prod()

    states = torch.zeros((args.num_steps, num_games, state_dim)).to(device)

    # Calculate occupancy dimension (which corresponds to your args.num_landmarks added during state processing)
    occupancy_dim = args.num_landmarks 
    n_max = args.num_landmarks * 2 # Or your intended max node structural capacity

    # Fix: Instantiate the correct class with all required positional arguments
    agent = GraphAgent(
        envs=envs, 
        n_max=n_max, 
        num_agents=num_agents_per_game, 
        occupancy_dim=occupancy_dim
    ).to(device)

    # Fix: Point the optimizer parameter groups to the real attributes inside GraphAgent
    optimizer = optim.Adam([
            {'params': list(agent.backbone.parameters()) + 
                       list(agent.actor_mlp.parameters()), 'lr': 3e-4}, 
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
            envs, _, _ = build_environments(args, run_name, new_seed, current_ratio, update_step=update)
            
            # Re-initialize the starting observations for PPO
            reset_data = envs.reset(seed=new_seed)
            if isinstance(reset_data, tuple):
                next_obs = torch.Tensor(reset_data[0]).to(device)
                next_info = reset_data[1]
            else:
                next_obs = torch.Tensor(reset_data).to(device)
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
            if "global_state" in next_info:
                current_game_states = torch.Tensor(next_info["global_state"][::num_agents_per_game]).to(device)
                raw_obs_tensor = torch.Tensor(next_info["raw_obs"]).to(device)
                landmark_dist = raw_obs_tensor.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
                landmark_dist = landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
                occupied = (torch.norm(landmark_dist, dim=-1) < strict_occupancy_radius).any(dim=1).float()
                states[step] = torch.cat([current_game_states, occupied], dim=-1)
            else:
                # Fallback for Atari: Reshape current observations as the "God-view"
                current_game_states = next_obs.view(num_games, -1)
                states[step] = current_game_states

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, centralized_state=states[step], denormalize=True)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            step_data = envs.step(action.cpu().numpy())
    
            # Gymnasium step returns (obs, reward, terminations, truncations, infos)
            if len(step_data) == 5:
                next_obs, reward, terminations, truncations, next_info = step_data
                done = terminations

                resets = np.logical_or(terminations, truncations)
            else:
                next_obs, reward, done, next_info = step_data
                resets = done

            if args.reward_cheat:
                agent_radius = 0.15
                collision_penalty = 0.5
                
                cheat_raw_obs = next_info["raw_obs"].copy()
                if "final_info" in next_info:
                    for i, fin_info in enumerate(next_info["final_info"]):
                        if fin_info is not None and "raw_obs" in fin_info:
                            cheat_raw_obs[i] = fin_info["raw_obs"]

                raw_obs_tensor = torch.Tensor(cheat_raw_obs).to(device)
                new_landmark_dist = raw_obs_tensor.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
                new_landmark_dist = new_landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
                
                # Shape: [num_games, num_agents, num_landmarks]
                dist = torch.norm(new_landmark_dist, dim=-1)
                
                # --- 1. EPISODIC STATIC ASSIGNMENT ---
                # Check which games just reset (or are at step 0)
                game_resets = torch.Tensor(resets).to(device).view(num_games, num_agents_per_game)[:, 0].bool()
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
                rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # LOGGING TEAM DATA
            if "final_info" in next_info:
                # Standard Gymnasium / SyncVectorEnv format
                for item in next_info["final_info"]:
                    if item is not None and "episode" in item:
                        r = item["episode"]["r"]
                        print(f"global_step={global_step}, episodic_return={r}")
                        writer.add_scalar("charts/team_episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/team_episodic_length", item["episode"]["l"], global_step)
                        break 
            elif "_episode" in next_info:
                # gym.wrappers.vector.RecordEpisodeStatistics format
                for i, done in enumerate(next_info["_episode"]):
                    if done:
                        r = next_info["episode"]["r"][i]
                        print(f"global_step={global_step}, episodic_return={r}")
                        # Extract the return from the arrays
                        writer.add_scalar("charts/team_episodic_return", next_info["episode"]["r"][i], global_step)
                        writer.add_scalar("charts/team_episodic_length", next_info["episode"]["l"][i], global_step)
                        break # Break after first item since all agents in one game share the reward

        # bootstrap value if not done
        with torch.no_grad():
            boot_obs = next_obs.clone()
            
            if "global_state" in next_info:
                boot_raw_obs = next_info["raw_obs"].copy()
                boot_global_state = next_info["global_state"].copy()
                
                # --- TELEPORTATION FIX ---
                # A. Inject Final Graph Obs
                if "final_observation" in next_info:
                    for agent_idx, final_obs in enumerate(next_info["final_observation"]):
                        if final_obs is not None:
                            boot_obs[agent_idx] = torch.Tensor(final_obs).to(device)
                
                # B. Inject Final Raw Obs & Global State
                if "final_info" in next_info:
                    for idx, fin_info in enumerate(next_info["final_info"]):
                        if fin_info is not None:
                            if "raw_obs" in fin_info:
                                boot_raw_obs[idx] = fin_info["raw_obs"]
                            if "global_state" in fin_info:
                                boot_global_state[idx] = fin_info["global_state"]

                # 2. Build the Final State correctly from TRUE Global State
                final_game_states = torch.Tensor(boot_global_state[::num_agents_per_game]).to(device)
                
                # 3. Calculate distances strictly from the RAW OBS!
                raw_obs_tensor = torch.Tensor(boot_raw_obs).to(device)
                landmark_dist = raw_obs_tensor.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
                landmark_dist = landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
                
                occupied = (torch.norm(landmark_dist, dim=-1) < strict_occupancy_radius).any(dim=1).float()
                final_state = torch.cat([final_game_states, occupied], dim=-1)
            else:
                # Fallback for Atari
                if "final_observation" in next_info:
                    for idx, final_obs in enumerate(next_info["final_observation"]):
                        if final_obs is not None:
                            boot_obs[idx] = torch.Tensor(final_obs).to(device)
                final_state = boot_obs.view(num_games, -1)

            # First Pass
            next_value = agent.get_value(boot_obs, centralized_state=final_state, denormalize=True).flatten()
    
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
            b_states = states.reshape((-1, state_dim))

            # 2. UPDATE STATS NOW (Before SGD)1
            # This aligns the normalizers with the data we just collected
            agent.critic_popart.update(b_returns.view(-1, agent.num_agents))

            agent.obs_normalizer.update(b_obs)

            # 3. Second Pass: RE-CALCULATE Values and Advantages with NEW stats
            # This is the crucial step you were missing. 
            # It ensures 'values' and 'returns' are in the same normalized space for SGD.
            new_values = agent.get_value(
                obs.view(-1, obs.shape[-1]), # FIX: Dynamically flatten based on tensor shape
                centralized_state=states.view(-1, state_dim), 
                denormalize=True
            ).view(args.num_steps, actual_num_envs)
            
            new_next_value = agent.get_value(boot_obs, centralized_state=final_state, denormalize=True).flatten()
            
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
        b_states = states.reshape((-1, state_dim))

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
                mb_states_for_critic = b_states[mb_state_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    centralized_state=mb_states_for_critic,
                    action=b_actions.long()[mb_inds]
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
                mb_returns_reshaped = b_returns[mb_inds].view(-1, agent.num_agents)
                mb_values_reshaped = b_values[mb_inds].view(-1, agent.num_agents)
                
                # Normalize targets correctly using the per-agent ID statistics
                normalized_returns = agent.critic_popart.normalize(mb_returns_reshaped).reshape(-1)
                normalized_values = agent.critic_popart.normalize(mb_values_reshaped).reshape(-1)

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
        writer.add_scalar("charts/local_ratio", current_ratio, global_step)
        writer.add_scalar("charts/ent_coef_now", ent_coef_now, global_step)

        if update % 100 == 0 or update == num_updates:
            save_dir = f"models/{run_name}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(agent.state_dict(), f"{save_dir}/{update}_model.pth")

    envs.close()
    writer.close()

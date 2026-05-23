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
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL_ma_il",
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

class Agent(nn.Module):
    def __init__(self, envs, num_agents, state_dim):
        super().__init__()
        self.num_agents = num_agents
        obs_shape = envs.single_observation_space.shape
        self.obs_dim = np.array(obs_shape).prod()
        # self.value_normalizer = ValueNormalizer(num_agents)
        # self.obs_normalizer = ObservationNormalizer(self.obs_dim)

        self.continuous_state_dim = state_dim - self.num_agents
        self.state_normalizer = ObservationNormalizer(self.continuous_state_dim)
        # self.state_normalizer = ObservationNormalizer(state_dim)

        # HETEROGENEOUS ACTORS: Each agent ID gets its own unique brain
        # self.actors = nn.ModuleList([
        #     nn.Sequential(
        #         layer_init(nn.Linear(self.obs_dim, 512)),
        #         nn.LayerNorm(512), nn.ReLU(),
        #         layer_init(nn.Linear(512, 512)),
        #         nn.LayerNorm(512), nn.ReLU(),
        #         layer_init(nn.Linear(512, 256)),
        #         nn.LayerNorm(256), nn.ReLU(),
        #         layer_init(nn.Linear(256, 256)),
        #         nn.ReLU(),
        #         layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        #     ) for _ in range(num_agents)
        # ])

        # Shared Actor
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 1024)),
            nn.LayerNorm(1024), nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.LayerNorm(1024), nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        )

        # SHARED CENTRALIZED CRITIC: One brain to judge the whole team
        concatenated_obs_dim = self.num_agents * self.obs_dim
        self.critic_projection = nn.Linear(concatenated_obs_dim, state_dim) if concatenated_obs_dim != state_dim else nn.Identity()
        
        self.critic_encoder = nn.Sequential(
            layer_init(nn.Linear(state_dim, 512)),
            nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.critic = PopArt(256, self.num_agents)

    def get_value(self, x, centralized_state=None, denormalize=False):
        if centralized_state is None:
            # x_norm = self.obs_normalizer.normalize(x)
            batch_size = x.shape[0]
            num_games = batch_size // self.num_agents
            proxy_state = x.view(num_games, -1)
            centralized_state = self.critic_projection(proxy_state)
        else:
            continuous_part = centralized_state[..., :-self.num_agents]
            binary_part = centralized_state[..., -self.num_agents:]
            
            # 2. Normalize only the continuous data
            norm_continuous = self.state_normalizer.normalize(continuous_part)
            
            # 3. Stitch them back together for the Critic network
            centralized_state = torch.cat([norm_continuous, binary_part], dim=-1)
        all_agent_values = self.critic(self.critic_encoder(centralized_state)) 
        if denormalize:
            all_agent_values = self.critic.denormalize(all_agent_values)
        return all_agent_values.view(-1, 1)

    # Shared Actor
    def get_action_and_value(self, x, action=None, centralized_state=None, denormalize=False):
        # x_norm = self.obs_normalizer.normalize(x)
        batch_size = x.shape[0]
        
        # Reshape to [NumGames, NumAgents, ObsDim]
        obs_reshaped = x.view(-1, self.num_agents, self.obs_dim)
        
        # VECTORIZED FORWARD PASS: Process all agents simultaneously
        logits = self.actor(obs_reshaped) 
        
        # Flatten back to [BatchSize, NumActions]
        combined_logits = logits.view(batch_size, -1)
        probs = Categorical(logits=combined_logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x, centralized_state, denormalize)
    
    # Heterogenous actors
    # def get_action_and_value(self, x, action=None, centralized_state=None, denormalize=False):
    #     x_norm = self.obs_normalizer.normalize(x)
    #     batch_size = x.shape[0]
        
    #     # Reshape to [NumGames, NumAgents, ObsDim]
    #     obs_reshaped = x_norm.view(-1, self.num_agents, self.obs_dim)
        
    #     logits_list = []
    #     for i in range(self.num_agents):
    #         # Each observation in the game is passed to its specific Actor
    #         logits = self.actors[i](obs_reshaped[:, i, :])
    #         logits_list.append(logits)
        
    #     # Flatten back to [BatchSize, NumActions]
    #     combined_logits = torch.stack(logits_list, dim=1).view(batch_size, -1)
    #     probs = Categorical(logits=combined_logits)
        
    #     if action is None:
    #         action = probs.sample()
            
    #     return action, probs.log_prob(action), probs.entropy(), self.get_value(x, centralized_state, denormalize)
    
    def load_bc_weights(self, bc_model_path="student_bc_best.pt"):
        bc_state_dict = torch.load(bc_model_path)
        actor_state_dict = {}
        
        # Strip the "network." prefix to align the dictionaries
        for key, value in bc_state_dict.items():
            if key.startswith("network."):
                new_key = key.replace("network.", "")
                actor_state_dict[new_key] = value
                
        self.actor.load_state_dict(actor_state_dict)
        print(f"\n[ORACLE] Successfully injected BC weights from {bc_model_path}!")

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

class NMaxObservationWrapper(BaseParallelWrapper):
    def __init__(self, env, n_max, current_n):
        super().__init__(env)
        self.n_max = n_max
        self.current_n = current_n
        self.target_dim = (4 * 6 * n_max) + n_max
        
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.target_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # Initialize locked slot placeholders
        self.landmark_slots = None
        self.agent_slots = None
        self.id_slots = None

    def reset(self, seed=None, options=None):
        # 1. LOCK THE PERMUTATION ONCE PER EPISODE
        self.landmark_slots = np.random.choice(self.n_max, self.current_n, replace=False)
        self.agent_slots = np.random.choice(self.n_max - 1, self.current_n - 1, replace=False)
        self.id_slots = np.random.choice(self.n_max, self.current_n, replace=False)

        obs, infos = self.env.reset(seed=seed, options=options)
        for agent in self.agents:
            infos[agent]["raw_obs"] = obs[agent]
        padded_obs = {agent: self.pad_obs(obs[agent]) for agent in obs}
        return padded_obs, infos
        
    def pad_obs(self, raw_obs):
        N = self.current_n
        frames_part = raw_obs[:-N]
        indicator_part = raw_obs[-N:]
        frames = frames_part.reshape(4, 6 * N)
        
        padded_frames = np.zeros((4, 6 * self.n_max), dtype=np.float32)
        padded_frames[:, 0:4] = frames[:, 0:4]
        
        landmarks = frames[:, 4:4+2*N].reshape(4, N, 2)
        
        # 2. USE THE LOCKED SLOTS (No more random generation here)
        for i, slot in enumerate(self.landmark_slots):
            idx_offset = 4 + (slot * 2)
            padded_frames[:, idx_offset:idx_offset+2] = landmarks[:, i, :]
            
        other_pos_start = 4 + 2*N
        other_pos = frames[:, other_pos_start:other_pos_start + 2*(N-1)].reshape(4, N-1, 2)
        other_vel_start = other_pos_start + 2*(N-1)
        other_vel = frames[:, other_vel_start:6*N].reshape(4, N-1, 2)
        
        for i, slot in enumerate(self.agent_slots):
            pos_idx = 4 + 2*self.n_max + (slot * 2)
            padded_frames[:, pos_idx:pos_idx+2] = other_pos[:, i, :]
            vel_idx = 4 + 2*self.n_max + 2*(self.n_max - 1) + (slot * 2)
            padded_frames[:, vel_idx:vel_idx+2] = other_vel[:, i, :]
            
        original_agent_idx = np.argmax(indicator_part)
        new_agent_idx = self.id_slots[original_agent_idx]
        
        padded_indicator = np.zeros(self.n_max, dtype=np.float32)
        padded_indicator[new_agent_idx] = 1.0
        
        padded_frames_flat = padded_frames.reshape(4 * 6 * self.n_max)
        return np.concatenate([padded_frames_flat, padded_indicator])
        
    def step(self, actions):
        obs, rews, terms, truncs, infos = self.env.step(actions)
        for agent in self.agents:
            infos[agent]["raw_obs"] = obs[agent] 
        padded_obs = {agent: self.pad_obs(obs[agent]) for agent in obs}
        return padded_obs, rews, terms, truncs, infos

# 1. Base Environment Factory
def make_env(args, env_id, seed, current_local_ratio=0.5):
    def thunk():
        if "simple_spread" in env_id:
            from mpe2 import simple_spread_v3
            env = simple_spread_v3.parallel_env(N=args.num_landmarks, max_cycles=args.max_cycles, 
                                                local_ratio=current_local_ratio, dynamic_rescaling=True, render_mode="rgb_array")
            env = StateWrapper(env)
        else:
            env = importlib.import_module(f"pettingzoo.atari.{env_id}").parallel_env(render_mode="rgb_array")
            env = ss.max_observation_v0(env, 2)
            env = ss.frame_skip_v0(env, 4)
            env = ss.color_reduction_v0(env, mode="B")
            env = ss.resize_v1(env, x_size=84, y_size=84)
            env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
        env = ss.frame_stack_v1(env, 4)
        env = ss.agent_indicator_v0(env, type_only=False)

        env = NMaxObservationWrapper(env, n_max=10, current_n=args.num_landmarks)
        
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
        # SuperSuit's concat_vec_envs returns a list of agent info dicts
        if isinstance(info, list):
            # Flatten the info for Gymnasium wrappers while preserving our custom key
            # Extract global_state from the first agent's info to the top level
            global_state_list = [i.get("global_state") for i in info]
            raw_obs_list = [i.get("raw_obs") for i in info]
            info = {
                "global_state": np.array(global_state_list), 
                "raw_obs": np.array(raw_obs_list),
                "agents": info
            }
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
            f"videos/{run_name}", 
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

    # reward_norm = RewardNormalizer(actual_num_envs, args.gamma)

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
        raw_obs_dim = len(next_info["raw_obs"][0])
        state_dim = (num_agents_per_game * raw_obs_dim) + args.num_landmarks
        # state_dim = (num_agents_per_game * np.array(envs.single_observation_space.shape).prod()) + args.num_landmarks

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

    agent = Agent(envs, num_agents_per_game, state_dim=state_dim).to(device)

    # from scratch optimizer
    # optimizer = optim.Adam([
    #         {'params': list(agent.actors.parameters()), 'lr': 3e-4}, 
    #         {'params': list(agent.critic_encoder.parameters()) + 
    #                 list(agent.critic.parameters()) + 
    #                 list(agent.critic_projection.parameters()), 'lr': 1e-3} 
    #     ], eps=1e-5)
    
    agent.load_bc_weights("student_bc_best_nmax10.pt")

    # behavorial clone optimizer
    optimizer = optim.Adam([
            {'params': list(agent.actor.parameters()), 'lr': 5e-5}, 
            {'params': list(agent.critic_encoder.parameters()) + 
                    list(agent.critic.parameters()) + 
                    list(agent.critic_projection.parameters()), 'lr': 1e-3} 
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
                if i == 0: # ACTOR
                    if update <= 50:
                        # PHASE 1: CRITIC WARMUP
                        # Freeze the Actor completely so the Critic can map the Value 
                        # of the expert BC weights without destroying them.
                        param_group["lr"] = 0.0
                    else:
                        # PHASE 2: GENTLE FINE-TUNING
                        # Unfreeze with the conservative learning rate
                        initial_lr = 1e-5 
                        param_group["lr"] = max(1e-6, frac * initial_lr)
                else: # CRITIC
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
                # current_game_states = torch.Tensor(next_info["global_state"][::num_agents_per_game]).to(device)
                raw_obs_tensor = torch.Tensor(next_info["raw_obs"]).to(device)

                current_game_states = raw_obs_tensor.view(num_games, -1)

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
                done = np.logical_or(terminations, truncations) # Combine for PPO
            else:
                next_obs, reward, done, next_info = step_data

            if args.reward_cheat:
                agent_radius = 0.15
                collision_penalty = 0.5
                
                next_obs_tensor = torch.Tensor(next_info["raw_obs"]).to(device)
                new_landmark_dist = next_obs_tensor.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
                new_landmark_dist = new_landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
                
                # Shape: [num_games, num_agents, num_landmarks]
                dist = torch.norm(new_landmark_dist, dim=-1)
                
                # --- 1. EPISODIC STATIC ASSIGNMENT ---
                # Check which games just reset (or are at step 0)
                game_dones = torch.Tensor(done).to(device).view(num_games, num_agents_per_game)[:, 0].bool()
                needs_assignment = needs_assignment | game_dones
                
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
            if "global_state" in next_info:
                # True global state for MPE
                # final_state = torch.Tensor(next_info["global_state"][::num_agents_per_game]).to(device)
                raw_obs_tensor = torch.Tensor(next_info["raw_obs"]).to(device)
                final_state = raw_obs_tensor.view(num_games, -1)

                landmark_dist = raw_obs_tensor.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
                landmark_dist = landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
                occupied = (torch.norm(landmark_dist, dim=-1) < strict_occupancy_radius).any(dim=1).float()
                final_state = torch.cat([final_state, occupied], dim=-1)
            else:
                # Fallback for Atari: Proxy state from observations
                final_state = next_obs.view(num_games, -1)

            next_value = agent.get_value(next_obs, centralized_state=final_state, denormalize=True).flatten()
    
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
            agent.critic.update(b_returns.view(-1, agent.num_agents))
            # agent.obs_normalizer.update(b_obs)

            continuous_b_states = b_states[:, :-args.num_landmarks]
            agent.state_normalizer.update(continuous_b_states)

            # 3. Second Pass: RE-CALCULATE Values and Advantages with NEW stats
            # This is the crucial step you were missing. 
            # It ensures 'values' and 'returns' are in the same normalized space for SGD.
            new_values = torch.zeros_like(values)
            for t in range(args.num_steps):
                # get_value now uses the updated normalization buffers
                new_values[t] = agent.get_value(obs[t], centralized_state=states[t], denormalize=True).flatten()
            
            new_next_value = agent.get_value(next_obs, centralized_state=final_state, denormalize=True).reshape(1, -1)
            
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
                    b_actions.long()[mb_inds],
                    centralized_state=mb_states_for_critic
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
                normalized_returns = agent.critic.normalize(mb_returns_reshaped).reshape(-1)
                normalized_values = agent.critic.normalize(mb_values_reshaped).reshape(-1)

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

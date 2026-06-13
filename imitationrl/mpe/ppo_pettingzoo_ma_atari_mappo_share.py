# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import argparse
import importlib
import os
import random
import time
import scipy
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
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL_ma",
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

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 512)),
            nn.LayerNorm(512), # Stabilizes coordinate inputs
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)), # Extra layer for coordination complexity
            nn.ReLU(),
        )
            
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

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
        else:
            env = importlib.import_module(f"pettingzoo.atari.{env_id}").parallel_env(render_mode="rgb_array")
            env = ss.max_observation_v0(env, 2)
            env = ss.frame_skip_v0(env, 4)
            env = ss.color_reduction_v0(env, mode="B")
            env = ss.resize_v1(env, x_size=84, y_size=84)
            env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
            env = ss.frame_stack_v1(env, 4) ### IMPORTANT ONLY FOR ATARI GAMES
        env = ss.agent_indicator_v0(env, type_only=False)
        
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
            # Safely extract true final observations if they exist
            final_obs_list = [i.get("final_observation", None) for i in info]
            info = {
                "global_state": np.array(global_state_list), 
                "final_obs": final_obs_list, # PRESERVE THIS
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
            info = {"global_state": np.array(global_state_list), "agents": info}
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
            f"videos_mlp/{run_name}", 
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
        state_dim = (num_agents_per_game * np.array(envs.single_observation_space.shape).prod()) + args.num_landmarks

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

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, actual_num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, actual_num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    dones = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    values = torch.zeros((args.num_steps, actual_num_envs)).to(device)
    ent_coef_now = 0

    current_assignments = torch.arange(args.num_landmarks).unsqueeze(0).repeat(num_games, 1).to(device)
    needs_assignment = torch.ones(num_games, dtype=torch.bool).to(device)

    for update in range(1, num_updates + 1):
    
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
                        
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

            needs_assignment.fill_(True)
            print("--- PHOENIX REBOOT COMPLETE ---")

        g_idx = torch.arange(num_games, device=device).view(-1, 1)
        a_idx = torch.arange(num_agents_per_game, device=device).view(1, -1)
        for step in range(0, args.num_steps):
            global_step += actual_num_envs
            obs[step] = next_obs
            dones[step] = next_done
            if "global_state" in next_info:
                # current_game_states = torch.Tensor(next_info["global_state"][::num_agents_per_game]).to(device)
                current_game_states = next_obs.view(num_games, -1)
                landmark_dist = next_obs.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
                landmark_dist = landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
                occupied = (torch.norm(landmark_dist, dim=-1) < strict_occupancy_radius).any(dim=1).float()
                states[step] = torch.cat([current_game_states, occupied], dim=-1)
            else:
                # Fallback for Atari: Reshape current observations as the "God-view"
                current_game_states = next_obs.view(num_games, -1)
                states[step] = current_game_states

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
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
                # CAST TO TENSOR FIRST
                next_obs_tensor = torch.Tensor(next_obs).to(device)

                # --- 1. REWARD CALCULATION FOR CURRENT STEP ---
                true_next_obs = next_obs_tensor.clone()
                if "final_obs" in next_info:
                    for agent_idx, final_obs in enumerate(next_info["final_obs"]):
                        if final_obs is not None:
                            true_next_obs[agent_idx] = torch.Tensor(final_obs).to(device)

                true_landmark_dist = true_next_obs.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
                true_landmark_dist = true_landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
                dist_for_reward = torch.norm(true_landmark_dist, dim=-1)

                assigned_dist = torch.gather(dist_for_reward, 2, current_assignments.unsqueeze(-1)).squeeze(-1)
                
                # --- 2. LOCALIZED REWARD ALGORITHMS ---
                individual_pull = -1.0 * assigned_dist
                
                landmarks_end = 4 + 2 * args.num_landmarks
                other_pos = true_next_obs.view(num_games, num_agents_per_game, -1)[:, :, landmarks_end:landmarks_end + 2 * (args.num_landmarks - 1)]
                other_pos = other_pos.view(num_games, num_agents_per_game, args.num_landmarks - 1, 2)
                other_dist = torch.norm(other_pos, dim=-1)
                
                collisions = (other_dist < 0.3).float().sum(dim=-1)
                individual_collision_penalty = -0.25 * collisions
                success_bonus = (assigned_dist < 0.1).float() * 1.0
                
                rewards[step] = individual_pull.view(-1) + individual_collision_penalty.view(-1) + success_bonus.view(-1)

                # --- 3. ASSIGNMENT ROUTING FOR NEXT EPISODE ---
                game_resets = torch.Tensor(resets).to(device).view(num_games, num_agents_per_game)[:, 0].bool()
                needs_assignment = needs_assignment | game_resets
                
                if needs_assignment.any():
                    reset_landmark_dist = next_obs_tensor.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
                    reset_landmark_dist = reset_landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
                    dist_for_new_episode = torch.norm(reset_landmark_dist, dim=-1)

                    dist_cpu = dist_for_new_episode.cpu().numpy()
                    
                    for g in range(num_games):
                        if needs_assignment[g]:
                            row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_cpu[g])
                            current_assignments[g] = torch.tensor(col_ind, device=device)
                                
                    needs_assignment.fill_(False)

                # --- 4. THE FIX: VECTORIZED OBSERVABILITY INJECTION ---
                # We physically re-order the landmarks so the Hungarian target is always FIRST,
                # but we do it using massive parallel tensor operations instead of slow Python loops.

                obs_reshaped = next_obs_tensor.clone().view(num_games, num_agents_per_game, -1)
                landmarks_segment = obs_reshaped[:, :, 4:4+2*args.num_landmarks].view(num_games, num_agents_per_game, args.num_landmarks, 2)

                # 1. Create a base index array of shape [num_games, num_agents_per_game, num_landmarks]
                # e.g., [0, 1, 2, 3] for every agent
                base_indices = torch.arange(args.num_landmarks, device=device).expand(num_games, num_agents_per_game, -1).clone()

                # 2. Perform the swap logically on the indices
                # Put the assigned target index into the 0th slot
                base_indices[:, :, 0] = current_assignments

                # Put 0 into the target's original slot using advanced grid indexing
                base_indices[g_idx, a_idx, current_assignments] = 0

                # 3. Expand the indices to cover the (X, Y) coordinate dimensions
                # Shape becomes: [num_games, num_agents_per_game, num_landmarks, 2]
                gather_indices = base_indices.unsqueeze(-1).expand(-1, -1, -1, 2)

                # 4. Fetch the swapped coordinates in one instantaneous GPU operation!
                swapped_landmarks = torch.gather(landmarks_segment, dim=2, index=gather_indices)

                # 5. Overwrite the observation block and flatten
                obs_reshaped[:, :, 4:4+2*args.num_landmarks] = swapped_landmarks.view(num_games, num_agents_per_game, -1)
                next_obs = obs_reshaped.view(actual_num_envs, -1)
            else:
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
            
            # 1. CREATE BOOTSTRAP_OBS FIRST
            bootstrap_obs = next_obs.clone() 
            if "final_obs" in next_info:
                for agent_idx, final_obs in enumerate(next_info["final_obs"]):
                    if final_obs is not None:
                        bootstrap_obs[agent_idx] = torch.Tensor(final_obs).to(device)

            # Get final values for bootstrapping
            next_value = agent.get_value(bootstrap_obs).flatten()
    
            # Standard GAE calculation
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values

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
        num_joint_steps = args.batch_size // num_agents_per_game
        joint_inds = np.arange(num_joint_steps)

        for epoch in range(args.update_epochs):
            rng.shuffle(joint_inds)
            for start in range(0, num_joint_steps, args.minibatch_size // num_agents_per_game):
                end = start + (args.minibatch_size // num_agents_per_game)
                # Pick joint indices and expand them to include all agents in those games
                mb_joint_inds = joint_inds[start:end]
                
                # This ensures we always pick Agent 0, 1, 2... from the same game/time together
                mb_inds = (mb_joint_inds[:, None] * num_agents_per_game + np.arange(num_agents_per_game)).flatten()

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    b_actions.long()[mb_inds]
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
                    mb_adv_reshaped = mb_advantages.view(-1, num_agents_per_game)
                    mb_adv_reshaped = (mb_adv_reshaped - mb_adv_reshaped.mean(dim=0)) / (mb_adv_reshaped.std(dim=0) + 1e-7)
                    mb_advantages = mb_adv_reshaped.reshape(-1)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

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

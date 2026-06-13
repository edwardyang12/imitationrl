# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool
import gc

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import vmas
import imageio

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL_ma_il")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="simple_spread")
    parser.add_argument("--total-timesteps", type=int, default=70000000)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--anneal-ent", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=0.03)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--num-landmarks", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=70)
    parser.add_argument("--reward-cheat", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
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

class Agent(nn.Module):
    def __init__(self, single_action_space, single_obs_shape, num_agents, state_dim):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = np.array(single_obs_shape).prod()
        self.continuous_state_dim = state_dim - self.num_agents
        self.state_normalizer = ObservationNormalizer(self.continuous_state_dim)

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
            layer_init(nn.Linear(512, single_action_space.n), std=0.01)
        )

        # SHARED CENTRALIZED CRITIC
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
            batch_size = x.shape[0]
            num_games = batch_size // self.num_agents
            proxy_state = x.view(num_games, -1)
            centralized_state = self.critic_projection(proxy_state)
        else:
            continuous_part = centralized_state[..., :-self.num_agents]
            binary_part = centralized_state[..., -self.num_agents:]
            
            norm_continuous = self.state_normalizer.normalize(continuous_part)
            centralized_state = torch.cat([norm_continuous, binary_part], dim=-1)
            
        all_agent_values = self.critic(self.critic_encoder(centralized_state)) 
        if denormalize:
            all_agent_values = self.critic.denormalize(all_agent_values)
        return all_agent_values.view(-1, 1)

    def get_action_and_value(self, x, action=None, centralized_state=None, denormalize=False):
        batch_size = x.shape[0]
        obs_reshaped = x.view(-1, self.num_agents, self.obs_dim)
        
        logits = self.actor(obs_reshaped) 
        combined_logits = logits.view(batch_size, -1)
        probs = Categorical(logits=combined_logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x, centralized_state, denormalize)
    
    def load_bc_weights(self, bc_model_path="student_bc_best.pt"):
        if not os.path.exists(bc_model_path):
            print(f"WARNING: BC model {bc_model_path} not found. Skipping.")
            return
        bc_state_dict = torch.load(bc_model_path)
        actor_state_dict = {}
        for key, value in bc_state_dict.items():
            if key.startswith("network."):
                actor_state_dict[key.replace("network.", "")] = value
        self.actor.load_state_dict(actor_state_dict)
        print(f"\n[ORACLE] Successfully injected BC weights from {bc_model_path}!")


# --- NATIVE PYTORCH VMAS ENVIRONMENT WRAPPER ---
# Replaces SuperSuit, ConcatVecEnv, DictInfoWrapper, and NMaxObservationWrapper
class VMASVectorizedEnv:
    def __init__(self, args, seed, run_name, update_step=0):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.num_agents = args.num_landmarks
        self.num_games = args.num_envs // self.num_agents
        self.num_envs = args.num_envs # Flat CleanRL convention
        self.run_name = run_name
        self.update_step = update_step
        
        # Native PyTorch Engine - Switched to simple_spread
        self.env = vmas.make_env(
            scenario="simple_spread",
            num_envs=self.num_games,
            device=self.device,
            continuous_actions=False,
            n_agents=self.num_agents,
            seed=seed,
            dict_spaces=False
        )
        
        self.single_action_space = gym.spaces.Discrete(5) # Discrete navigation
        
        # MPE simple_spread native dimension is exactly 6 * N
        # [vel(2), pos(2), lm_rel(2*N), other_pos(2N-2), comms(2N-2)]
        self.mpe_base_dim = 6 * self.num_agents
        
        # Stacked 4 times + One-Hot Agent Indicator(N)
        target_dim = (self.mpe_base_dim * 4) + self.num_agents
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(target_dim,))
        
        # GPU Frame Stack Buffer [Games, Agents, 4_frames, Obs_Dim]
        self.frames = torch.zeros((self.num_games, self.num_agents, 4, self.mpe_base_dim), device=self.device)
        
        self.step_count = 0
        self.video_frames = []

    def _vmas_to_mpe_obs(self, vmas_obs):
        # VMAS's simple_spread is a direct port of MPE's simple_spread.
        # Natively matches the [vel, pos, entity_pos, other_pos, comm] structure
        stacked = torch.stack(vmas_obs, dim=1) 
        
        # Safely pad/truncate to 6*N to guarantee structural integrity with your pretrained weights
        if stacked.shape[-1] == self.mpe_base_dim:
            return stacked
        else:
            mpe_obs = torch.zeros((self.num_games, self.num_agents, self.mpe_base_dim), device=self.device)
            actual_dim = min(stacked.shape[-1], self.mpe_base_dim)
            mpe_obs[:, :, :actual_dim] = stacked[:, :, :actual_dim]
            return mpe_obs

    def _apply_stack_and_indicator(self, raw_mpe_obs):
        # Fast GPU Tensor Shift for frame stacking
        self.frames = torch.roll(self.frames, shifts=-1, dims=2)
        self.frames[:, :, -1, :] = raw_mpe_obs
        
        # Flatten the 4 frames 
        stacked_flat = self.frames.view(self.num_games, self.num_agents, -1)
        
        # Create persistent identity matrix for one-hot encoding [Games, Agents, Num_Agents]
        indicators = torch.eye(self.num_agents, device=self.device).unsqueeze(0).expand(self.num_games, -1, -1)
            
        final_obs = torch.cat([stacked_flat, indicators], dim=-1)
        return final_obs.reshape(self.num_envs, -1)

    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
            
        vmas_obs = self.env.reset()
        self.frames.zero_()
        self.step_count = 0
        
        mpe_obs = self._vmas_to_mpe_obs(vmas_obs)
        for _ in range(4):
            final_obs = self._apply_stack_and_indicator(mpe_obs)
            
        info = {
            "raw_obs": mpe_obs.reshape(self.num_envs, -1),
            "global_state": final_obs # Satisfies the info["global_state"] check in the main loop
        }
        return final_obs, info

    def step(self, actions):
        # Translate flat CleanRL actions -> Native VMAS continuous tensor structure
        actions_reshaped = actions.view(self.num_games, self.num_agents)
        action_list = [actions_reshaped[:, i] for i in range(self.num_agents)]
        
        vmas_obs, vmas_rews, _, vmas_info = self.env.step(action_list)
        self.step_count += 1
        
        # Native VMAS RGB Rendering Logic
        if self.args.capture_video and self.step_count % self.args.max_cycles == 0:
            frame = self.env.render(mode="rgb_array", no_window=True, agent_index_focus=None)
            if isinstance(frame, list): frame = frame[0] # Record first game only
            self.video_frames.append(frame)

        mpe_obs = self._vmas_to_mpe_obs(vmas_obs)
        final_obs = self._apply_stack_and_indicator(mpe_obs)
        
        # Vectorize Returns
        rewards = torch.stack(vmas_rews, dim=1).reshape(-1)
        is_done = self.step_count >= self.args.max_cycles
        dones = torch.full((self.num_envs,), is_done, device=self.device, dtype=torch.float32)
        
        info = {
            "raw_obs": mpe_obs.reshape(self.num_envs, -1),
            "global_state": final_obs
        }
        
        if is_done:
            if self.args.capture_video and self.video_frames:
                os.makedirs(f"videos/{self.run_name}", exist_ok=True)
                imageio.mimsave(f"videos/{self.run_name}/rl-video-update_{self.update_step}.mp4", self.video_frames, fps=15)
                self.video_frames = []

            # Gym Return Tracking Equivalent
            info["_episode"] = [True] * self.num_envs
            info["episode"] = {
                "r": (rewards * self.args.max_cycles).cpu().numpy(),
                "l": [self.args.max_cycles] * self.num_envs
            }
            # Auto-Reset
            vmas_obs = self.env.reset()
            self.frames.zero_()
            self.step_count = 0
            mpe_obs = self._vmas_to_mpe_obs(vmas_obs)
            for _ in range(4):
                final_obs = self._apply_stack_and_indicator(mpe_obs)
                
        return final_obs, rewards, dones, info

    def close(self):
        pass


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
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    strict_occupancy_radius = 0.2

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = VMASVectorizedEnv(args, args.seed, run_name, update_step=0)
    actual_num_envs = envs.num_envs
    num_agents_per_game = envs.num_agents
    num_games = envs.num_games

    args.batch_size = int(actual_num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    num_updates = args.total_timesteps // args.batch_size    

    global_step = 0
    start_time = time.time()
    
    # Native Tensor Initializers
    next_obs, next_info = envs.reset(seed=args.seed)
    next_done = torch.zeros(actual_num_envs, device=device)

    # State config parsing
    state_dim = (num_agents_per_game * np.array(envs.single_observation_space.shape).prod()) + args.num_landmarks

    states = torch.zeros((args.num_steps, num_games, state_dim), device=device)
    agent = Agent(envs.single_action_space, envs.single_observation_space.shape, num_agents_per_game, state_dim=state_dim).to(device)
    
    agent.load_bc_weights("student_bc_best_nmax10.pt")

    optimizer = optim.Adam([
            {'params': list(agent.actor.parameters()), 'lr': 5e-5}, 
            {'params': list(agent.critic_encoder.parameters()) + 
                    list(agent.critic.parameters()) + 
                    list(agent.critic_projection.parameters()), 'lr': 1e-3} 
        ], eps=1e-5)

    obs = torch.zeros((args.num_steps, actual_num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, actual_num_envs), device=device) # Removed single_action_space.shape to match discrete
    logprobs = torch.zeros((args.num_steps, actual_num_envs), device=device)
    rewards = torch.zeros((args.num_steps, actual_num_envs), device=device)
    dones = torch.zeros((args.num_steps, actual_num_envs), device=device)
    values = torch.zeros((args.num_steps, actual_num_envs), device=device)
    ent_coef_now = 0

    for update in range(1, num_updates + 1):
        current_assignments = torch.arange(args.num_landmarks, device=device).unsqueeze(0).repeat(num_games, 1)
        needs_assignment = torch.ones(num_games, dtype=torch.bool, device=device)
        
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0: 
                    if update <= 50: param_group["lr"] = 0.0
                    else: param_group["lr"] = max(5e-6, frac * 5e-5)
                else: 
                    param_group["lr"] = max(1e-4, frac * 1e-3)
                
        if args.anneal_ent:
            progress = (update - 1.0) / num_updates
            if progress < 0.7: ent_coef_now = args.ent_coef 
            else:
                decay_progress = (progress - 0.7) / 0.3
                ent_coef_now = max(0.0001, args.ent_coef * (1.0 - decay_progress))
        else:
            ent_coef_now = args.ent_coef

        if update % 100 == 0:
            envs.close()
            new_seed = args.seed + update 
            envs = VMASVectorizedEnv(args, new_seed, run_name, update_step=update)
            next_obs, next_info = envs.reset(seed=new_seed)
            next_done = torch.zeros(actual_num_envs, device=device)

        for step in range(0, args.num_steps):
            global_step += actual_num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            raw_obs_tensor = next_info["raw_obs"]
            current_game_states = next_obs.view(num_games, -1)

            landmark_dist = raw_obs_tensor.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
            landmark_dist = landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
            occupied = (torch.norm(landmark_dist, dim=-1) < strict_occupancy_radius).any(dim=1).float()
            states[step] = torch.cat([current_game_states, occupied], dim=-1)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, centralized_state=states[step], denormalize=True)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # NATIVE TENSOR EXECUTION - Removing .cpu().numpy() bottlenecks
            next_obs, reward, done, next_info = envs.step(action)

            if args.reward_cheat:
                agent_radius = 0.15
                collision_penalty = 0.5
                
                next_obs_tensor = next_info["raw_obs"]
                new_landmark_dist = next_obs_tensor.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
                new_landmark_dist = new_landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
                dist = torch.norm(new_landmark_dist, dim=-1)
                
                game_dones = done.view(num_games, num_agents_per_game)[:, 0].bool()
                needs_assignment = needs_assignment | game_dones
                
                if needs_assignment.any():
                    dist_clone = dist.clone()
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
                
                assigned_dist = torch.gather(dist, 2, current_assignments.unsqueeze(-1)).squeeze(-1)
                min_snap_multiplier = collision_penalty / np.sqrt(agent_radius)
                snap_factor = 1.5 
                procedural_multiplier = min_snap_multiplier * snap_factor
                individual_pull = -procedural_multiplier * torch.sqrt(assigned_dist + 1e-8)
                rewards[step] = reward.view(-1) + individual_pull.view(-1)
            else:
                rewards[step] = reward.view(-1)
                
            next_done = done

            if "_episode" in next_info:
                for i, is_terminated in enumerate(next_info["_episode"]):
                    if is_terminated:
                        r = next_info["episode"]["r"][i]
                        print(f"global_step={global_step}, episodic_return={r}")
                        writer.add_scalar("charts/team_episodic_return", next_info["episode"]["r"][i], global_step)
                        writer.add_scalar("charts/team_episodic_length", next_info["episode"]["l"][i], global_step)
                        break 

        with torch.no_grad():
            raw_obs_tensor = next_info["raw_obs"]
            final_state = next_obs.view(num_games, -1)
            landmark_dist = raw_obs_tensor.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks]
            landmark_dist = landmark_dist.view(num_games, num_agents_per_game, args.num_landmarks, 2)
            occupied = (torch.norm(landmark_dist, dim=-1) < strict_occupancy_radius).any(dim=1).float()
            final_state = torch.cat([final_state, occupied], dim=-1)

            next_value = agent.get_value(next_obs, centralized_state=final_state, denormalize=True).flatten()
    
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
            b_states = states.reshape((-1, state_dim))

            agent.critic.update(b_returns.view(-1, agent.num_agents))
            continuous_b_states = b_states[:, :-args.num_landmarks]
            agent.state_normalizer.update(continuous_b_states)

            new_values = torch.zeros_like(values)
            for t in range(args.num_steps):
                new_values[t] = agent.get_value(obs[t], centralized_state=states[t], denormalize=True).flatten()
            
            new_next_value = agent.get_value(next_obs, centralized_state=final_state, denormalize=True).reshape(1, -1)
            
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
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_states = states.reshape((-1, state_dim))

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        num_joint_steps = args.batch_size // agent.num_agents
        joint_inds = np.arange(num_joint_steps)

        for epoch in range(args.update_epochs):
            rng.shuffle(joint_inds)
            for start in range(0, num_joint_steps, args.minibatch_size // agent.num_agents):
                end = start + (args.minibatch_size // agent.num_agents)
                mb_joint_inds = joint_inds[start:end]
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
                mb_returns_reshaped = b_returns[mb_inds].view(-1, agent.num_agents)
                mb_values_reshaped = b_values[mb_inds].view(-1, agent.num_agents)
                
                normalized_returns = agent.critic.normalize(mb_returns_reshaped).reshape(-1)
                normalized_values = agent.critic.normalize(mb_values_reshaped).reshape(-1)

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

        if update % 100 == 0 or update == num_updates:
            save_dir = f"models/{run_name}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(agent.state_dict(), f"{save_dir}/{update}_model.pth")

    envs.close()
    writer.close()
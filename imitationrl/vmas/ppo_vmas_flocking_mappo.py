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
# from torch.distributions.categorical import Categorical
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
    parser.add_argument("--wandb-project-name", type=str, default="vmas",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="flocking",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=7e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=2048,
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
            
        # 3. Strict Index-Preserved Teammates
        # We iterate over ALL agents so the array structure never shifts.
        # When a == agent, the relative pos/vel becomes [0, 0], which the network ignores.
        for a in self.world.agents:
            obs.append(a.state.pos - agent.state.pos)
            obs.append(a.state.vel - agent.state.vel) 
                
        # 4. Relative Obstacles
        for landmark in self.world.landmarks:
            if landmark != self._target:
                obs.append(landmark.state.pos - agent.state.pos)
                
        return torch.cat(obs, dim=-1)

    # def reward(self, agent):
    #     # 1. Get the base flocking reward (cohesion, alignment, target tracking)
    #     base_reward = super().reward(agent)
        
    #     obstacle_penalty = torch.zeros_like(base_reward)
        
    #     # 2. Calculate continuous vectorized penalties for obstacles
    #     for landmark in self.world.landmarks:
    #         if landmark != self._target:
    #             dist = torch.linalg.vector_norm(agent.state.pos - landmark.state.pos, dim=-1)
    #             collision_dist = agent.shape.radius + landmark.shape.radius
                
    #             # --- CONTINUOUS SHAPING ---
    #             # Create a "danger zone" slightly larger than the collision radius
    #             margin = 0.15
    #             danger_zone = collision_dist + margin
                
    #             # If inside the danger zone, calculate a smooth penalty that increases as they get closer.
    #             # Max soft penalty is -0.05 right at the collision boundary.
    #             soft_penalty = torch.where(
    #                 dist < danger_zone, 
    #                 -0.05 * ((danger_zone - dist) / margin), 
    #                 0.0
    #             )
                
    #             # Apply a smaller hard penalty (-0.05 instead of -1.0) for actual physical intersection
    #             hard_penalty = torch.where(dist < collision_dist, -0.05, 0.0)
                
    #             obstacle_penalty += (soft_penalty + hard_penalty)
                
    #     # 3. Combine and return
    #     return base_reward + obstacle_penalty

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
    def __init__(self, single_action_space, single_obs_shape, num_agents, state_dim):
        super().__init__()
        self.num_agents = num_agents

        self.obs_dim = np.array(single_obs_shape).prod()
        self.action_dim = np.prod(single_action_space.shape)
        # self.value_normalizer = ValueNormalizer(num_agents)
        self.obs_normalizer = ObservationNormalizer(self.obs_dim)

        self.agent_id_embedding = nn.Embedding(num_agents, 10) # this 10 can be changed later

        self.state_normalizer = ObservationNormalizer(state_dim)

        # SHARED ACTOR: One brain for all agents
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim + 10, 1024)),
            nn.LayerNorm(1024), nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )

        # The Mean Head (Restored Tanh for instant braking)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(256, self.action_dim), std=0.01),
            nn.Tanh() 
        )

        # The Variance Head (Starts with a bias of -0.5 to match your original initialization)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

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
            x_norm = self.obs_normalizer.normalize(x)
            batch_size = x.shape[0]
            num_games = batch_size // self.num_agents
            proxy_state = x_norm.view(num_games, -1)
            centralized_state = self.critic_projection(proxy_state)
        else:
            centralized_state = self.state_normalizer.normalize(centralized_state)
            
        all_agent_values = self.critic(self.critic_encoder(centralized_state)) 
        if denormalize:
            all_agent_values = self.critic.denormalize(all_agent_values)
        return all_agent_values.view(-1, 1)

    def get_action_and_value(self, x, action=None, centralized_state=None, denormalize=False):
        x_norm = self.obs_normalizer.normalize(x)
        batch_size = x.shape[0] 
        
        agent_ids = torch.arange(self.num_agents).to(x.device).repeat(batch_size // self.num_agents)
        embedded_ids = self.agent_id_embedding(agent_ids)
        actor_input = torch.cat([x_norm, embedded_ids], dim=-1)
        
        # Extract features
        actor_features = self.actor(actor_input)
        
        # Branch into Mean and LogStd
        action_means = self.actor_mean(actor_features)
        action_logstds = self.actor_logstd.expand_as(action_means)
        
        # Loosely bound the logstd to prevent NaN explosions, allowing near-zero variance
        safe_logstds = torch.clamp(action_logstds, min=-5.0, max=2.0)
        action_stds = safe_logstds.exp()
        
        probs = Normal(action_means, action_stds)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x, centralized_state, denormalize)

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
            scenario= FlockingCleanObs() if args.env_id == "flocking" else args.env_id,
            num_envs=self.num_games,
            device=self.device,
            continuous_actions=True,
            n_agents=self.num_agents,
            seed=seed,
            dict_spaces=False
        )
        
        vmas_obs_dim = self.env.observation_space[0].shape[0]
        self.single_action_space = self.env.action_space[0]
        
        target_dim = vmas_obs_dim + self.num_agents
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(target_dim,))
        
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        
        self.step_count = 0
        self.episode_count = 0
        self.record_this_episode = self.args.capture_video
        self.video_frames = []

    def _apply_stack_and_indicator(self, stacked_vmas_obs):
        indicators = torch.eye(self.num_agents, device=self.device).unsqueeze(0).expand(self.num_games, -1, -1)
        final_obs = torch.cat([stacked_vmas_obs, indicators], dim=-1)
        return final_obs.reshape(self.num_envs, -1)

    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
            
        vmas_obs = self.env.reset()
        self.episode_returns.zero_()
        self.step_count = 0
        
        stacked_obs = torch.stack(vmas_obs, dim=1)
        final_obs = self._apply_stack_and_indicator(stacked_obs)
            
        info = {
            "raw_obs": stacked_obs.reshape(self.num_envs, -1),
            "global_state": final_obs 
        }
        return final_obs, info

    def step(self, actions):
        actions_reshaped = actions.view(self.num_games, self.num_agents, -1)
        
        # Pass continuous force vectors directly to VMAS
        vmas_actions = [actions_reshaped[:, i, :] for i in range(self.num_agents)]
        
        vmas_obs, vmas_rews, _, vmas_info = self.env.step(vmas_actions)
        self.step_count += 1
        
        rewards = torch.stack(vmas_rews, dim=1).reshape(-1)
        self.episode_returns += rewards

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

        stacked_obs = torch.stack(vmas_obs, dim=1)
        final_obs = self._apply_stack_and_indicator(stacked_obs)
        
        is_done = self.step_count >= self.args.max_cycles
        dones = torch.full((self.num_envs,), is_done, device=self.device, dtype=torch.float32)
        
        info = {
            "raw_obs": stacked_obs.reshape(self.num_envs, -1),
            "global_state": final_obs.clone()
        }
        
        if is_done:
            self.episode_count += 1
            
            info["terminal_raw_obs"] = info["raw_obs"].clone()
            info["terminal_global_state"] = info["global_state"].clone()
            
            if self.record_this_episode and self.video_frames:
                os.makedirs(f"videos/{self.run_name}", exist_ok=True)
                file_path = f"videos/{self.run_name}/rl-video-update_{self.update_step}-ep_{self.episode_count}.mp4"
                imageio.mimsave(file_path, self.video_frames, fps=15)
                self.video_frames = []
            
            if self.args.capture_video and self.episode_count % 50 == 0:
                self.record_this_episode = True
            else:
                self.record_this_episode = False

            # Surface the completed returns before wiping them
            info["episode"] = {
                "r": self.episode_returns.clone(),
                "l": torch.full((self.num_envs,), self.step_count, device=self.device)
            }

            vmas_obs = self.env.reset()
            self.episode_returns.zero_()
            self.step_count = 0
            
            stacked_obs = torch.stack(vmas_obs, dim=1)
            final_obs = self._apply_stack_and_indicator(stacked_obs)
            
            info["raw_obs"] = stacked_obs.reshape(self.num_envs, -1)
            info["global_state"] = final_obs.clone()
                
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

    states = torch.zeros((args.num_steps, num_games, state_dim)).to(device)

    agent = Agent(envs.single_action_space, envs.single_observation_space.shape, num_agents_per_game, state_dim=state_dim).to(device)
    optimizer = optim.Adam([
            {'params': list(agent.actor.parameters()) + 
                       list(agent.actor_mean.parameters()) + 
                       [agent.actor_logstd] + 
                       list(agent.agent_id_embedding.parameters()), 'lr': 3e-4}, 
            
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
            current_game_states = next_obs.view(num_games, -1)
            states[step] = current_game_states

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, centralized_state=states[step], denormalize=True)
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

                resets = np.logical_or(terminations, truncations)
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
                rewards[step] = reward.clone().to(device).view(-1)
            next_obs, next_done = next_obs.clone().to(device), done.clone().to(device)

            # LOGGING TEAM DATA
            if done.any() and "episode" in next_info:
                done_bool = done.bool()
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
            
            if "global_state" in next_info:
                # 1. Create a separate tracker for the Raw MPE physics
                boot_raw_obs = next_info["raw_obs"].clone()
                
                # --- TELEPORTATION FIX ---
                # A. Inject Final Graph Obs
                if "terminal_global_state" in next_info:
                    boot_obs = next_info["terminal_global_state"].clone()
                
                if "terminal_raw_obs" in next_info:
                    boot_raw_obs = next_info["terminal_raw_obs"].clone()

                # 2. Build the Final State correctly
                final_state = boot_obs.view(num_games, -1)
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
            agent.critic.update(b_returns.view(-1, agent.num_agents))

            agent.obs_normalizer.update(b_obs)
            agent.state_normalizer.update(b_states)

            # 3. Second Pass: RE-CALCULATE Values and Advantages with NEW stats
            # This is the crucial step you were missing. 
            # It ensures 'values' and 'returns' are in the same normalized space for SGD.
            new_values = agent.get_value(
                obs.view(-1, agent.obs_dim), 
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
                    b_actions[mb_inds], 
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
        writer.add_scalar("charts/ent_coef_now", ent_coef_now, global_step)

        if update % 100 == 0 or update == num_updates:
            save_dir = f"models/{run_name}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(agent.state_dict(), f"{save_dir}/{update}_model.pth")

    envs.close()
    writer.close()

    print("--- STARTING LONG INFERENCE EVALUATION ---")
    
    # 1. Lock the agent's normalizers (CRITICAL)
    agent.eval() 
    
    eval_max_cycles = 1500 # Set this to however long you want to watch
    
    # 2. Spin up a fresh, single-game environment with the long max-cycles
    eval_env = vmas.make_env(
        scenario=FlockingCleanObs() if args.env_id == "flocking" else args.env_id,
        num_envs=1, # Just one game for the video
        device=device,
        continuous_actions=True,
        n_agents=num_agents_per_game,
        seed=args.seed + 100, # Offset seed so it's a novel starting position
        dict_spaces=False
    )
    
    # 3. Reset and prep the observation format
    obs_list = eval_env.reset()
    stacked_obs = torch.stack(obs_list, dim=1)
    
    # Recreate the indicator concatenation you do in your wrapper
    indicators = torch.eye(num_agents_per_game, device=device).unsqueeze(0)
    next_obs = torch.cat([stacked_obs, indicators], dim=-1).reshape(num_agents_per_game, -1).to(device)
    
    frames = []

    # 4. The rollout loop
    with torch.no_grad():
        for step in range(eval_max_cycles):
            # Get action (stochastic, just like training)
            action, _, _, _ = agent.get_action_and_value(next_obs)
            clipped_action = torch.clamp(action, -1.0, 1.0)
            
            # Reshape for VMAS engine [num_games, num_agents, action_dim]
            actions_reshaped = clipped_action.view(1, num_agents_per_game, -1)
            vmas_actions = [actions_reshaped[:, i, :] for i in range(num_agents_per_game)]
            
            # Step the environment
            vmas_obs, _, _, _ = eval_env.step(vmas_actions)
            
            # Capture the frame
            frame = eval_env.render(mode="rgb_array", env_index=0)
            if isinstance(frame, list): 
                frame = frame[0]
            frames.append(frame)
            
            # Process next observation
            stacked_obs = torch.stack(vmas_obs, dim=1)
            next_obs = torch.cat([stacked_obs, indicators], dim=-1).reshape(num_agents_per_game, -1).to(device)

    
    os.makedirs(f"videos/{run_name}", exist_ok=True)
    video_path = f"videos/{run_name}/FINAL_LONG_EVAL_{eval_max_cycles}_cycles.mp4"
    imageio.mimsave(video_path, frames, fps=15)
    print(f"--- LONG EVALUATION SAVED TO {video_path} ---")

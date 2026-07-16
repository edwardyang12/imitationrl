import argparse
import os
import gc
import torch
import numpy as np
import imageio
from tqdm import tqdm
import json

# Import the architecture and environment wrapper directly from your training script
# from ppo_vmas_navigation_gnn import GraphAgent, VMASVectorizedEnv
from ppo_vmas_navigation_mappo import Agent, VMASVectorizedEnv

class BehavioralMetricTracker:
    def __init__(self, num_games, num_agents, agent_radius=0.15, contact_threshold=0.30, goal_tolerance=0.15):
        self.num_games = num_games
        self.num_agents = num_agents
        self.agent_radius = agent_radius
        self.contact_threshold = contact_threshold
        self.goal_tolerance = goal_tolerance
        self.reset()

    def reset(self):
        # Existing Spatial Accumulators
        self.total_steps = 0
        self.collision_events = 0
        self.total_agent_steps = 0
        self.free_speeds = []
        self.congested_speeds = []
        self.min_clearances = []
        self.action_jitters = []
        
        # New Diagnostic Accumulators
        self.deadlock_events = 0
        self.energy_expenditures = []
        self.final_goal_distances = None
        
        # Path Tortuosity anchors
        self.start_positions = None
        self.initial_goal_distances = None
        self.distance_traveled = None
        self.prev_positions = None
        self.prev_actions = None

    def update(self, raw_obs_flat, actions_flat):
        raw_obs = raw_obs_flat.view(self.num_games, self.num_agents, -1)
        pos = raw_obs[:, :, 0:2]
        vel = raw_obs[:, :, 2:4]
        to_goal = raw_obs[:, :, 4:6]
        actions = actions_flat.view(self.num_games, self.num_agents, -1)
        
        # Always track latest goal distance for final convergence rate
        goal_dists = torch.norm(to_goal, dim=-1) # [B, N]
        self.final_goal_distances = goal_dists.clone()
        
        if self.start_positions is None:
            self.start_positions = pos.clone()
            self.prev_positions = pos.clone()
            self.initial_goal_distances = goal_dists.clone()
            self.distance_traveled = torch.zeros((self.num_games, self.num_agents), device=pos.device)
            self.prev_actions = actions.clone()
            return

        self.total_steps += 1
        self.total_agent_steps += (self.num_games * self.num_agents)
        
        # 1. Pairwise Distances & Collisions (C_rate, d_min)
        pos_i = pos.unsqueeze(2)
        pos_j = pos.unsqueeze(1)
        dist_matrix = torch.norm(pos_i - pos_j, dim=-1)
        mask = torch.eye(self.num_agents, device=pos.device).bool().unsqueeze(0)
        dist_matrix.masked_fill_(mask, float('inf'))
        
        closest_dist, _ = dist_matrix.min(dim=-1)
        self.min_clearances.append(closest_dist.mean().item())
        self.collision_events += (closest_dist < self.contact_threshold).sum().item()
        
        # 2. Velocity Degradation (V_deg)
        speeds = torch.norm(vel, dim=-1)
        is_congested = closest_dist < (self.agent_radius * 4.0)
        if is_congested.any():
            self.congested_speeds.append(speeds[is_congested].mean().item())
        if (~is_congested).any():
            self.free_speeds.append(speeds[~is_congested].mean().item())
            
        # 3. NEW: Deadlock / Freeze Frequency (F_rate)
        # Agent is stationary (< 0.05 m/s) BUT still far from goal (> goal_tolerance)
        is_deadlocked = (speeds < 0.05) & (goal_dists > self.goal_tolerance)
        self.deadlock_events += is_deadlocked.sum().item()
        
        # 4. NEW: Mechanical Energy Expenditure (E)
        action_norms = torch.norm(actions, dim=-1)
        self.energy_expenditures.append(action_norms.mean().item())
            
        # 5. Tortuosity (tau) & Control Jitter (J)
        self.distance_traveled += torch.norm(pos - self.prev_positions, dim=-1)
        self.prev_positions = pos.clone()
        
        action_diff = torch.norm(actions - self.prev_actions, dim=-1)
        self.action_jitters.append(action_diff.mean().item())
        self.prev_actions = actions.clone()

    def get_summary(self):
        valid_goals = self.initial_goal_distances > 0.1
        tortuosity = (self.distance_traveled[valid_goals] / self.initial_goal_distances[valid_goals]).mean().item() if (valid_goals is not None and valid_goals.any()) else 1.0
        
        v_free = np.mean(self.free_speeds) if self.free_speeds else 1e-5
        v_cong = np.mean(self.congested_speeds) if self.congested_speeds else 0.0
        
        # Calculate Goal Convergence Rate at the final recorded step
        success_rate = 0.0
        if self.final_goal_distances is not None:
            converged = (self.final_goal_distances < self.goal_tolerance).float()
            success_rate = (converged.mean().item()) * 100.0
        
        return {
            "S_rate (Goal Convergence Rate %)": round(success_rate, 2),
            "F_rate (Deadlock Freeze Frequency %)": round((self.deadlock_events / max(1, self.total_agent_steps)) * 100, 2),
            "C_rate (Collision Frequency %)": round((self.collision_events / max(1, self.total_agent_steps)) * 100, 2),
            "d_min (Mean Minimum Clearance m)": round(float(np.mean(self.min_clearances)), 3) if self.min_clearances else 0.0,
            "V_deg (Velocity Degradation Ratio)": round(float(v_cong / max(1e-5, v_free)), 3),
            "Tau (Trajectory Tortuosity)": round(float(tortuosity), 3),
            "E_bar (Mean Mechanical Energy / Force)": round(float(np.mean(self.energy_expenditures)), 3) if self.energy_expenditures else 0.0,
            "J (Action Control Jitter)": round(float(np.mean(self.action_jitters)), 4) if self.action_jitters else 0.0
        }

def parse_harvest_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved .pth oracle model file")
    parser.add_argument("--num-landmarks", type=int, default=7, help="Number of agents/landmarks to harvest (N)")
    parser.add_argument("--n-max", type=int, default=5, help="MUST MATCH TRAINING: The context window size")
    parser.add_argument("--num-trajectories", type=int, default=500000, help="Total step transitions to harvest")
    parser.add_argument("--chunk-size", type=int, default=100000, help="How many steps to hold in RAM before writing to disk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the harvesting environment")
    parser.add_argument("--output-dir", type=str, default="./expert_data", help="Directory to save the harvested numpy arrays")
    parser.add_argument("--video-interval", type=int, default=10000, help="Record a sample video every X steps")
    parser.add_argument("--max-cycles", type=int, default=250, help="Length of an environment episode before auto-reset")
    
    args = parser.parse_args()
    
    # Inject hidden args required by VMASVectorizedEnv initialization
    args.cuda = torch.cuda.is_available()
    args.env_id = "navigation"
    args.capture_video = False 
    args.reward_cheat = False
    
    # --- FIX THE BUG HERE ---
    # Even though we are harvesting up to --num-trajectories continuously,
    # the env internal step counter needs this to flag environment Resets.
    # We grab it from args.max_cycles or manually inject it if you remove the parser argument.
    
    # Vectorize environments based on the population size 
    args.num_envs = args.num_landmarks 
    
    return args

def harvest_vmas_dataset():
    args = parse_harvest_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    video_dir = os.path.join(args.output_dir, "expert_videos")
    os.makedirs(video_dir, exist_ok=True)

    video_frames = []
    is_recording = False
    
    print(f"--- INITIALIZING DETERMINISTIC EXPERT HARVESTING ---")
    print(f"Oracle Model: {args.model_path}")
    print(f"Population Size (N): {args.num_landmarks}")
    print(f"Target Steps: {args.num_trajectories} (Chunks of {args.chunk_size})")
    
    # 1. Initialize VMAS Environment
    envs = VMASVectorizedEnv(args, args.seed, run_name="harvest_run", update_step=0)
    video_tracker = BehavioralMetricTracker(envs.num_games, envs.num_agents)
    
    # 2. Initialize and Load Oracle Agent
    n_max_nodes = args.n_max * 2

    # MLP
    state_dim = envs.num_agents * np.array(envs.single_observation_space.shape).prod()
    oracle = Agent(
        envs.single_action_space, 
        envs.single_observation_space.shape, 
        num_agents = envs.num_agents, 
        state_dim=state_dim, 
        n_max=n_max_nodes
    ).to(device)

    # oracle = GraphAgent(
    #     envs=envs, 
    #     n_max=n_max_nodes, 
    #     num_agents=args.num_landmarks
    # ).to(device)
    
    print("Loading Oracle weights (filtering out training-only Critic)...")
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    
    # 1. Delete all critic and embedding keys from the dictionary
    keys_to_remove = [k for k in state_dict.keys() if "critic" in k or "embedding" in k]
    for k in keys_to_remove:
        del state_dict[k]
        
    # 2. Load only the matching Actor weights safely!
    oracle.load_state_dict(state_dict, strict=False)
    oracle.eval()
    
    # Data buffers
    expert_obs = []
    expert_actions = []
    chunk_idx = 0
    
    # Initial Environment Reset
    reset_data = envs.reset(seed=args.seed)
    if isinstance(reset_data, tuple):
        obs = reset_data[0].clone().to(device)
        raw_obs = reset_data[1]["raw_obs"].clone().to(device) # <- Cleanly extract raw physical coords
    else:
        obs = reset_data.clone().to(device)
        raw_obs = obs.clone()
    
    print("Starting data collection...")
    with torch.no_grad():
        for step in tqdm(range(args.num_trajectories)):

            if step % args.video_interval == 0:
                is_recording = True
                video_frames = []

                video_tracker.reset()
                print(f"\n[Video] Starting recording at step {step}...")

            # --- DETERMINISTIC FORWARD PASS ---
            # 1. Normalize the incoming observations if an obs_normalizer exists
            if hasattr(oracle, 'obs_normalizer'):
                obs_norm = oracle.obs_normalizer.normalize(obs)
            else:
                obs_norm = obs
            
            # 2. Pass through the GNN backbone to get valid nodes and embeddings
            # (Matches the backbone signature: returns valid_x, node_embeddings, and optionally batch/etc.)
            if hasattr(oracle, 'backbone'):
                backbone_outputs = oracle.backbone(obs_norm)
                valid_x = backbone_outputs[0]
                node_embeddings = backbone_outputs[1]
                
                # 3. Filter for active ego-agents taking actions
                agent_mask = valid_x[:, 4] > 0.5
                
                # 4. Extract MLP features and map directly to deterministic action means (no sampling!)
                actor_features = oracle.actor_mlp(node_embeddings[agent_mask])
            else:
                # MLP Forward Pass (Append agent ID embeddings)
                actor_features = oracle.actor(obs_norm)
            deterministic_action = oracle.actor_mean(actor_features)
            
            # Clip actions to valid physical simulation limits [-1, 1]
            noise = torch.randn_like(deterministic_action) * 0.025
            clipped_action = torch.clamp(deterministic_action + noise, -1.0, 1.0)

            # --- RENDER FRAME IF RECORDING ---
            if is_recording:
                # Render the frame directly from the underlying VMAS engine

                video_tracker.update(raw_obs, clipped_action)

                frame = envs.env.render(mode="rgb_array", env_index=0, agent_index_focus=None)
                if isinstance(frame, list):
                    frame = frame[0]
                video_frames.append(frame)
                
                # If we've collected enough frames, save the video file
                if len(video_frames) >= args.max_cycles:
                    video_path = os.path.join(video_dir, f"expert_step_{step}.mp4")
                    metrics_path = os.path.join(video_dir, f"expert_step_{step}_metrics.json")
                    imageio.mimsave(video_path, video_frames, fps=15)

                    episode_metrics = video_tracker.get_summary()
                    with open(metrics_path, "w") as f:
                        json.dump(episode_metrics, f, indent=2)
                        
                    print(f"\n[Video & Metrics] Saved sample behavior to:\n  -> {video_path}\n  -> {metrics_path}")
                    print(f"  -> Episode C_rate: {episode_metrics['C_rate (Collision Frequency %)']}% | d_min: {episode_metrics['d_min (Mean Minimum Clearance m)']}m")
                    video_frames = []
                    is_recording = False
            
            # 5. Append un-pushed raw observations and continuous actions to RAM logs
            expert_obs.append(obs.cpu().numpy())
            expert_actions.append(clipped_action.cpu().numpy())
            
            # 6. Environment Step execution
            step_data = envs.step(clipped_action)
            obs = step_data[0].clone().to(device)

            if len(step_data) >= 4 and isinstance(step_data[-1], dict) and "raw_obs" in step_data[-1]:
                raw_obs = step_data[-1]["raw_obs"].clone().to(device)
            
            # 7. Memory check & safe array dumping
            if len(expert_obs) >= args.chunk_size:
                obs_array = np.vstack(expert_obs)
                act_array = np.vstack(expert_actions)
                
                obs_save_path = os.path.join(args.output_dir, f"obs_N{args.num_landmarks}_part{chunk_idx}.npy")
                act_save_path = os.path.join(args.output_dir, f"actions_N{args.num_landmarks}_part{chunk_idx}.npy")
                
                np.save(obs_save_path, obs_array)
                np.save(act_save_path, act_array)
                print(f"\n[Memory Check] Saved chunk {chunk_idx}. Flushing RAM...")
                
                expert_obs.clear()
                expert_actions.clear()
                gc.collect()
                chunk_idx += 1
                
    # Save residual data elements left over after loop conclusion
    if len(expert_obs) > 0:
        obs_array = np.vstack(expert_obs)
        act_array = np.vstack(expert_actions)
        
        np.save(os.path.join(args.output_dir, f"obs_N{args.num_landmarks}_part{chunk_idx}.npy"), obs_array)
        np.save(os.path.join(args.output_dir, f"actions_N{args.num_landmarks}_part{chunk_idx}.npy"), act_array)
        print(f"\n[Memory Check] Saved final remainder chunk to part{chunk_idx}.")
        
    print(f"Successfully finished harvesting N={args.num_landmarks}!")
    envs.close()

if __name__ == "__main__":
    harvest_vmas_dataset()
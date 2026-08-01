import argparse
import os
import gc
import csv
import torch
import numpy as np
import imageio
from tqdm import tqdm
import json

# Import the architecture and environment wrapper directly from your training script
# from ppo_vmas_navigation_gnn import GraphAgent, VMASVectorizedEnv
from ppo_vmas_navigation_mappo import Agent, TransformerAgent, PointNetAgent, VMASVectorizedEnv

class BehavioralMetricTracker:
    def __init__(self, num_games, num_agents, agent_radius=0.1, contact_threshold=0.20, goal_tolerance=0.25):
        self.num_games = num_games
        self.num_agents = num_agents
        self.agent_radius = agent_radius
        self.contact_threshold = contact_threshold
        self.goal_tolerance = goal_tolerance
        self.reset()

    def reset(self):
        # Spatial Accumulators
        self.total_steps = 0
        self.collision_events = 0
        self.total_active_agent_steps = 0 
        self.free_speeds = []
        self.congested_speeds = []
        self.min_clearances = []
        
        # Active Phase Tracking
        self.action_jitters = []
        self.energy_expenditures = []
        
        # Settled (Post-Arrival) Phase Tracking
        self.post_arrival_jitters = []
        self.post_arrival_energy = []
        self.max_settled_energy = 0.0
        
        # Yield & Spike Tracking
        self.currently_at_goal = None
        self.cooperative_yields = 0
        self.forced_displacements = 0
        
        # Diagnostic Accumulators
        self.deadlock_events = 0
        self.final_goal_distances = None
        
        # Active vs Idle State Tracking
        self.done_mask = None
        self.convergence_steps = None
        
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
        
        goal_dists = torch.norm(to_goal, dim=-1) # [B, N]
        self.final_goal_distances = goal_dists.clone()
        
        if self.start_positions is None:
            self.start_positions = pos.clone()
            self.prev_positions = pos.clone()
            self.initial_goal_distances = goal_dists.clone()
            self.distance_traveled = torch.zeros((self.num_games, self.num_agents), device=pos.device)
            self.prev_actions = actions.clone()
            
            # Initialize state tracking tensors
            self.done_mask = torch.zeros((self.num_games, self.num_agents), dtype=torch.bool, device=pos.device)
            self.convergence_steps = torch.full((self.num_games, self.num_agents), float('inf'), device=pos.device)
            self.currently_at_goal = goal_dists < self.goal_tolerance
            return

        self.total_steps += 1
        
        # --- CALCULATE GLOBAL DISTANCES FIRST ---
        # We need this for all agents (not just active ones) to detect bumps
        pos_i = pos.unsqueeze(2)
        pos_j = pos.unsqueeze(1)
        dist_matrix = torch.norm(pos_i - pos_j, dim=-1)
        mask = torch.eye(self.num_agents, device=pos.device).bool().unsqueeze(0)
        dist_matrix.masked_fill_(mask, float('inf'))
        closest_dist, _ = dist_matrix.min(dim=-1)

        # --- ADVANCED YIELD TRACKING ---
        at_goal_now = goal_dists < self.goal_tolerance
        if self.currently_at_goal is not None:
            just_departed = self.currently_at_goal & (~at_goal_now)
            
            if just_departed.any():
                step_jitters = torch.norm(actions - self.prev_actions, dim=-1)
                
                # Condition 1: Was it collision-free?
                is_collision_free = closest_dist >= self.contact_threshold
                
                # Condition 2: Was it a smooth, intentional movement?
                # A threshold of 0.5 filters out violent left/right thrashing 
                is_intentional = step_jitters < 0.5 
                
                # Filter the departures
                true_yields = just_departed & is_collision_free & is_intentional
                forced_bumps = just_departed & (~(is_collision_free & is_intentional))
                
                self.cooperative_yields += true_yields.sum().item()
                self.forced_displacements += forced_bumps.sum().item()
        
        self.currently_at_goal = at_goal_now.clone()
        
        # 1. Update Done Mask and freeze completion times
        just_finished = (goal_dists < self.goal_tolerance) & (~self.done_mask)
        self.convergence_steps[just_finished] = self.total_steps
        
        # Latch the done mask (once true, stays true)
        self.done_mask = self.done_mask | (goal_dists < self.goal_tolerance)
        
        # Split masks
        active_mask = ~self.done_mask
        settled_mask = self.done_mask
        
        active_count = active_mask.sum().item()
        self.total_active_agent_steps += active_count
        
        # --- SETTLED PHASE METRICS ---
        if settled_mask.any():
            action_norms = torch.norm(actions, dim=-1)
            step_jitters = torch.norm(actions - self.prev_actions, dim=-1)
            
            # Track the absolute highest energy spike of any settled agent
            current_max_energy = action_norms[settled_mask].max().item()
            self.max_settled_energy = max(self.max_settled_energy, current_max_energy)
            
            self.post_arrival_energy.append(action_norms[settled_mask].mean().item())
            self.post_arrival_jitters.append(step_jitters[settled_mask].mean().item())

        # --- ACTIVE PHASE METRICS ---
        if active_count > 0:
            active_closest = closest_dist[active_mask]
            
            self.min_clearances.append(active_closest.mean().item())
            self.collision_events += (active_closest < self.contact_threshold).sum().item()
            
            # Velocity Degradation
            speeds = torch.norm(vel, dim=-1)
            active_speeds = speeds[active_mask]
            active_congested_mask = active_closest < (self.agent_radius * 4.0)
            
            if active_congested_mask.any():
                self.congested_speeds.append(active_speeds[active_congested_mask].mean().item())
            if (~active_congested_mask).any():
                self.free_speeds.append(active_speeds[~active_congested_mask].mean().item())
                
            # Deadlock Frequency
            is_deadlocked = (active_speeds < 0.05) & (goal_dists[active_mask] > self.goal_tolerance)
            self.deadlock_events += is_deadlocked.sum().item()
            
            # Active Energy and Jitter
            action_norms = torch.norm(actions, dim=-1)
            step_jitters = torch.norm(actions - self.prev_actions, dim=-1)
            
            self.energy_expenditures.append(action_norms[active_mask].mean().item())
            self.action_jitters.append(step_jitters[active_mask].mean().item())
                
            # Tortuosity
            step_distances = torch.norm(pos - self.prev_positions, dim=-1)
            self.distance_traveled[active_mask] += step_distances[active_mask]
        
        # Update previous states
        self.prev_positions = pos.clone()
        self.prev_actions = actions.clone()

    def get_summary(self):
        valid_goals = self.initial_goal_distances > 0.1
        
        # 1. T_conv: Based strictly on the first time they arrived
        converged_mask = self.convergence_steps < float('inf')
        mean_t_conv = self.convergence_steps[converged_mask].mean().item() if converged_mask.any() else self.total_steps
        
        # 2. S_rate: Based strictly on the final frame retention
        success_rate = 0.0
        if self.final_goal_distances is not None:
            # Did they actually hold the landmark at the very end of the episode?
            retained = (self.final_goal_distances < self.goal_tolerance).float()
            success_rate = (retained.mean().item()) * 100.0

        # 3. Tortuosity: Restricted to ONLY agents that at least reached the goal once
        valid_tortuosity = valid_goals & converged_mask
        tortuosity = (self.distance_traveled[valid_tortuosity] / self.initial_goal_distances[valid_tortuosity]).mean().item() if valid_tortuosity.any() else 1.0
        
        v_free = np.mean(self.free_speeds) if self.free_speeds else 1e-5
        v_cong = np.mean(self.congested_speeds) if self.congested_speeds else 0.0
        
        return {
            "S_rate (Final Goal Retention Rate %)": round(success_rate, 2),
            "T_conv (Mean Steps to First Arrival)": round(float(mean_t_conv), 1),
            "Yields_Cooperative": self.cooperative_yields,
            "Yields_Forced_Displacement": self.forced_displacements,
            "F_rate (Active Deadlock Frequency %)": round((self.deadlock_events / max(1, self.total_active_agent_steps)) * 100, 2),
            "C_rate (Active Collision Frequency %)": round((self.collision_events / max(1, self.total_active_agent_steps)) * 100, 2),
            "d_min (Active Min Clearance m)": round(float(np.mean(self.min_clearances)), 3) if self.min_clearances else 0.0,
            "V_deg (Active Velocity Degradation)": round(float(v_cong / max(1e-5, v_free)), 3),
            "Tau (Active Trajectory Tortuosity)": round(float(tortuosity), 3),
            "E_active (Active Mean Energy)": round(float(np.mean(self.energy_expenditures)), 3) if self.energy_expenditures else 0.0,
            "J_active (Active Control Jitter)": round(float(np.mean(self.action_jitters)), 4) if self.action_jitters else 0.0,
            "E_settled_mean (Post-Arrival Mean Energy)": round(float(np.mean(self.post_arrival_energy)), 3) if self.post_arrival_energy else 0.0,
            "E_settled_max (Peak Yielding Force)": round(float(self.max_settled_energy), 3),
            "J_settled (Post-Arrival Control Jitter)": round(float(np.mean(self.post_arrival_jitters)), 4) if self.post_arrival_jitters else 0.0
        }

def parse_harvest_args():
    parser = argparse.ArgumentParser()
    
    # --- Mode Configuration ---
    parser.add_argument("--mode", type=str, choices=["harvest", "inference"], default="harvest", 
                        help="Choose 'harvest' for long data collection or 'inference' for a multi-N evaluation sweep.")
    parser.add_argument("--prefix", type=str, default="", 
                        help="An optional string to prepend to the output video folder and CSV filename.")
    
    # --- Inference Arguments ---
    parser.add_argument("--n-test-array", type=int, nargs="+", default=[5, 7, 10], 
                        help="List of N (num agents) to test during inference mode (e.g., --n-test-array 5 10 15).")
    parser.add_argument("--csv-output", type=str, default="inference_metrics.csv", 
                        help="Path to save the inference results CSV.")
    
    # --- Standard Harvest Arguments ---
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved .pth oracle model file")
    parser.add_argument("--num-landmarks", type=int, default=7, help="Number of agents/landmarks to harvest (N) for harvest mode")
    parser.add_argument("--n-max", type=int, default=5, help="MUST MATCH TRAINING: The context window size")
    parser.add_argument("--num-trajectories", type=int, default=500000, help="Total step transitions to harvest")
    parser.add_argument("--chunk-size", type=int, default=100000, help="How many steps to hold in RAM before writing to disk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the harvesting environment")
    parser.add_argument("--output-dir", type=str, default="./expert_data", help="Directory to save the harvested numpy arrays")
    parser.add_argument("--video-interval", type=int, default=10000, help="Record a sample video every X steps")
    parser.add_argument("--max-cycles", type=int, default=350, help="Length of an environment episode before auto-reset")
    
    args = parser.parse_args()
    
    args.cuda = torch.cuda.is_available()
    args.env_id = "navigation"
    args.capture_video = False 
    args.reward_cheat = False
    args.num_envs = args.num_landmarks 
    
    return args

def load_oracle_model(args, envs, device):
    """Helper to initialize architecture and load strict weights."""
    state_dim = envs.num_agents * np.array(envs.single_observation_space.shape).prod()
    
    # oracle = TransformerAgent(
    #     envs.single_action_space, 
    #     envs.single_observation_space.shape, 
    #     envs.num_agents, 
    #     state_dim=state_dim, 
    #     n_max=args.n_max * 2
    # ).to(device)

    # oracle = Agent(
    #     envs.single_action_space, 
    #     envs.single_observation_space.shape, 
    #     num_agents = envs.num_agents, 
    #     state_dim=state_dim, 
    #     n_max=args.n_max * 2
    # ).to(device)

    # oracle = GraphAgent(
    #     envs=envs, 
    #     n_max=args.n_max * 2, 
    #     num_agents=args.num_landmarks
    # ).to(device)

    oracle = PointNetAgent(
        envs.single_action_space, 
        envs.single_observation_space.shape, 
        num_agents = envs.num_agents, 
        state_dim=state_dim, 
        n_max=args.n_max * 2
    ).to(device)

    # MLP and GraphAgent alternatives remain functionally available here if uncommented
    
    print(f"Loading Oracle weights from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    
    keys_to_remove = [k for k in state_dict.keys() if "critic" in k or "embedding" in k]
    for k in keys_to_remove:
        del state_dict[k]
        
    oracle.load_state_dict(state_dict, strict=False)
    oracle.eval()
    return oracle

def get_action(oracle, obs):
    """Helper to perform deterministic forward pass across environments."""
    if hasattr(oracle, 'obs_normalizer'):
        obs_norm = oracle.obs_normalizer.normalize(obs)
    else:
        obs_norm = obs
        
    if hasattr(oracle, 'backbone'):
        backbone_outputs = oracle.backbone(obs_norm)
        valid_x = backbone_outputs[0]
        node_embeddings = backbone_outputs[1]
        agent_mask = valid_x[:, 4] > 0.5
        actor_features = oracle.actor_mlp(node_embeddings[agent_mask])
    elif hasattr(oracle, 'transformer'):
        backbone_outputs = oracle._forward_actor_backbone(obs_norm)
        actor_features = oracle.actor(backbone_outputs)
    elif hasattr(oracle, 'rho'):
        # PointNet routing uses rho instead of actor
        backbone_outputs = oracle._forward_actor_backbone(obs_norm)
        actor_features = oracle.rho(backbone_outputs)
    else:
        actor_features = oracle.actor(obs_norm)
        
    deterministic_action = oracle.actor_mean(actor_features)
    noise = torch.randn_like(deterministic_action) * 0.025
    clipped_action = torch.clamp(deterministic_action + noise, -1.0, 1.0)
    
    return clipped_action

def run_inference(args):
    """Runs exactly one episode for each N in --n-test-array, saves a video, and logs metrics to CSV."""
    device = torch.device("cuda" if args.cuda else "cpu")
    all_metrics = []
    
    # Handle the prefix formatting
    prefix_str = f"{args.prefix}_" if args.prefix else ""
    
    # Create a directory specifically for inference videos
    video_folder_name = f"{prefix_str}inference_videos"
    video_dir = os.path.join(args.output_dir, video_folder_name)
    os.makedirs(video_dir, exist_ok=True)
    
    # Resolve the final CSV path
    csv_dirname = os.path.dirname(args.csv_output)
    csv_basename = os.path.basename(args.csv_output)
    final_csv_name = f"{prefix_str}{csv_basename}"
    final_csv_path = os.path.join(csv_dirname, final_csv_name) if csv_dirname else final_csv_name
    
    print(f"\n--- STARTING DETERMINISTIC INFERENCE SWEEP ---")
    print(f"Oracle Model: {args.model_path}")
    print(f"Prefix: '{args.prefix}'" if args.prefix else "Prefix: None")
    print(f"N_test configurations to evaluate: {args.n_test_array}")
    
    for n_test in args.n_test_array:
        print(f"\nEvaluating N = {n_test}...")
        
        # Override arguments dynamically for the current N_test
        args.num_landmarks = n_test
        args.num_envs = n_test
        
        envs = VMASVectorizedEnv(args, args.seed, run_name=f"{prefix_str}infer_run_{n_test}", update_step=0)
        tracker = BehavioralMetricTracker(envs.num_games, envs.num_agents)
        oracle = load_oracle_model(args, envs, device)
        
        reset_data = envs.reset(seed=args.seed)
        if isinstance(reset_data, tuple):
            obs = reset_data[0].clone().to(device)
            raw_obs = reset_data[1]["raw_obs"].clone().to(device)
        else:
            obs = reset_data.clone().to(device)
            raw_obs = obs.clone()
            
        video_frames = []
            
        with torch.no_grad():
            for step in tqdm(range(args.max_cycles), desc=f"Episode Progress (N={n_test})"):
                action = get_action(oracle, obs)
                
                tracker.update(raw_obs, action)
                
                # Render the frame for this step
                frame = envs.env.render(mode="rgb_array", env_index=0, agent_index_focus=None)
                if isinstance(frame, list):
                    frame = frame[0]
                video_frames.append(frame)
                
                step_data = envs.step(action)
                obs = step_data[0].clone().to(device)
                if len(step_data) >= 4 and isinstance(step_data[-1], dict) and "raw_obs" in step_data[-1]:
                    raw_obs = step_data[-1]["raw_obs"].clone().to(device)

        # Save the video for this N_test
        video_path = os.path.join(video_dir, f"{prefix_str}inference_N{n_test}.mp4")
        imageio.mimsave(video_path, video_frames, fps=15)
        print(f"Saved video to: {video_path}")

        metrics = tracker.get_summary()
        # Prepend the N parameter configuration to the dict
        metrics = {"N_test": n_test, **metrics}
        all_metrics.append(metrics)
        
        print(f"Results for N={n_test}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
            
        envs.close()

    # Aggregate and Save to CSV
    if all_metrics:
        os.makedirs(os.path.dirname(os.path.abspath(final_csv_path)), exist_ok=True)
        keys = all_metrics[0].keys()
        
        with open(final_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_metrics)
            
        print(f"\n[Success] All inference metrics saved to: {final_csv_path}")

def harvest_imitation_data(args):
    """Runs long-term data collection up to --num-trajectories."""
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
    
    envs = VMASVectorizedEnv(args, args.seed, run_name="harvest_run", update_step=0)
    video_tracker = BehavioralMetricTracker(envs.num_games, envs.num_agents)
    oracle = load_oracle_model(args, envs, device)
    
    expert_obs = []
    expert_actions = []
    chunk_idx = 0
    
    reset_data = envs.reset(seed=args.seed)
    if isinstance(reset_data, tuple):
        obs = reset_data[0].clone().to(device)
        raw_obs = reset_data[1]["raw_obs"].clone().to(device) 
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

            clipped_action = get_action(oracle, obs)

            if is_recording:
                video_tracker.update(raw_obs, clipped_action)
                frame = envs.env.render(mode="rgb_array", env_index=0, agent_index_focus=None)
                if isinstance(frame, list):
                    frame = frame[0]
                video_frames.append(frame)
                
                if len(video_frames) >= args.max_cycles:
                    video_path = os.path.join(video_dir, f"expert_step_{step}.mp4")
                    metrics_path = os.path.join(video_dir, f"expert_step_{step}_metrics.json")
                    imageio.mimsave(video_path, video_frames, fps=15)

                    episode_metrics = video_tracker.get_summary()
                    with open(metrics_path, "w") as f:
                        json.dump(episode_metrics, f, indent=2)
                        
                    print(f"\n[Video & Metrics] Saved sample behavior to:\n  -> {video_path}\n  -> {metrics_path}")
                    video_frames = []
                    is_recording = False
            
            expert_obs.append(obs.cpu().numpy())
            expert_actions.append(clipped_action.cpu().numpy())
            
            step_data = envs.step(clipped_action)
            obs = step_data[0].clone().to(device)
            if len(step_data) >= 4 and isinstance(step_data[-1], dict) and "raw_obs" in step_data[-1]:
                raw_obs = step_data[-1]["raw_obs"].clone().to(device)
            
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
                
    if len(expert_obs) > 0:
        obs_array = np.vstack(expert_obs)
        act_array = np.vstack(expert_actions)
        np.save(os.path.join(args.output_dir, f"obs_N{args.num_landmarks}_part{chunk_idx}.npy"), obs_array)
        np.save(os.path.join(args.output_dir, f"actions_N{args.num_landmarks}_part{chunk_idx}.npy"), act_array)
        print(f"\n[Memory Check] Saved final remainder chunk to part{chunk_idx}.")
        
    print(f"Successfully finished harvesting N={args.num_landmarks}!")
    envs.close()

if __name__ == "__main__":
    args = parse_harvest_args()
    
    if args.mode == "harvest":
        harvest_imitation_data(args)
    elif args.mode == "inference":
        run_inference(args)
import argparse
import os
import gc
import csv
import torch
import numpy as np
import imageio
from tqdm import tqdm
import math
import json

# Import the architecture, environment wrapper, and metrics directly from your transport training script
from ppo_vmas_transport_mappo import PointNetAgent, MAPPOAgent, VMASVectorizedEnv
# from ppo_vmas_transport_gnn import GraphAgent, VMASVectorizedEnv

def compute_transport_metrics(world, agent_radius=0.03):
    B = len(world.agents[0].state.pos)
    N = len(world.agents)
    
    agent_pos = torch.stack([a.state.pos for a in world.agents], dim=1) 
    agent_vel = torch.stack([a.state.vel for a in world.agents], dim=1) 
    package_pos = torch.stack([l.state.pos for l in world.landmarks if l.movable], dim=1) 
    package_vel = torch.stack([l.state.vel for l in world.landmarks if l.movable], dim=1) 
    goal_pos = torch.stack([l.state.pos for l in world.landmarks if not l.movable and not l.collide], dim=1) 
    
    # 1. Package-to-Goal Transport Error & Velocity Goodput
    if package_pos.shape[1] > 0 and goal_pos.shape[1] > 0:
        # Calculate exact distances for every individual package
        package_dists = torch.norm(package_pos - goal_pos, dim=-1) # Shape: (B, P)
        
        # Transport error remains the mean distance for smooth gradient/charting
        transport_error = package_dists.mean(dim=1)
        
        to_goal_dir = goal_pos - package_pos
        to_goal_dist = package_dists.unsqueeze(-1) + 1e-8
        to_goal_unit = to_goal_dir / to_goal_dist
        
        transport_velocity = (package_vel * to_goal_unit).sum(dim=-1).mean(dim=1)
        
        # FIX: Success is True only if ALL packages are within the collision boundary
        # Goal radius (0.15) + Box half-width (0.175) = 0.325 boundary intersection.
        success_rate = (package_dists < 0.398).float().mean(dim=1)
    else:
        transport_error = torch.zeros(B, device=agent_pos.device)
        transport_velocity = torch.zeros(B, device=agent_pos.device)
        success_rate = torch.zeros(B, device=agent_pos.device)
        
    if package_pos.shape[1] > 0:
        dist_to_pkgs = torch.norm(agent_pos.unsqueeze(2) - package_pos.unsqueeze(1), dim=-1)
        min_dist_to_pkg, _ = torch.min(dist_to_pkgs, dim=-1)
        contact_distance = min_dist_to_pkg.mean(dim=1)
        
        # New: Active Engagement Rate
        contact_threshold = 0.30 
        is_engaged = (min_dist_to_pkg < contact_threshold).float()
        engagement_rate = is_engaged.mean(dim=1)
    else:
        contact_distance = torch.zeros(B, device=agent_pos.device)
        engagement_rate = torch.zeros(B, device=agent_pos.device)
        is_engaged = torch.zeros((B, N), device=agent_pos.device)
        
    # Collision Rate
    pairwise_distances = torch.norm(agent_pos.unsqueeze(2) - agent_pos.unsqueeze(1), dim=-1)
    pairwise_distances.masked_fill_(torch.eye(N, device=agent_pos.device).unsqueeze(0).bool(), float('inf'))
    collision_rate = (pairwise_distances < (agent_radius * 2.0)).any(dim=-1).float().mean(dim=1)

    # New: Push Alignment (Force Vectoring)
    speeds = torch.norm(agent_vel, dim=-1, keepdim=True) + 1e-8
    headings = agent_vel / speeds
    active_headings = headings * is_engaged.unsqueeze(-1)
    active_count = is_engaged.sum(dim=1, keepdim=True) + 1e-8
    push_alignment = torch.norm(active_headings.sum(dim=1) / active_count, dim=-1)
    push_alignment = push_alignment * (is_engaged.sum(dim=1) > 0).float()
    
    return {
        "transport_error": transport_error,
        "transport_velocity": transport_velocity,
        "contact_distance": contact_distance,
        "collision_rate": collision_rate,
        "engagement_rate": engagement_rate,
        "push_alignment": push_alignment,
        "success_rate": success_rate
    }

class TransportMetricTracker:
    def __init__(self, num_games, num_agents):
        self.num_games = num_games
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        self.total_steps = 0
        
        # Accumulate sums and active step counts PER GAME
        self.active_steps = torch.zeros(self.num_games)
        self.transport_error_sum = torch.zeros(self.num_games)
        self.transport_velocity_sum = torch.zeros(self.num_games)
        self.contact_distance_sum = torch.zeros(self.num_games)
        self.collision_rate_sum = torch.zeros(self.num_games)
        self.engagement_rate_sum = torch.zeros(self.num_games)
        self.push_alignment_sum = torch.zeros(self.num_games)

        # Completion tracking
        self.is_completed = torch.zeros(self.num_games, dtype=torch.bool)
        self.completion_times = torch.full((self.num_games,), float('nan'))

        self.max_success_rate = torch.zeros(self.num_games)

    def update(self, world):
        metrics = compute_transport_metrics(world)
        device = metrics["success_rate"].device
        
        # Ensure tracking tensors are on the same device as the physics engine
        self.is_completed = self.is_completed.to(device)
        self.completion_times = self.completion_times.to(device)
        self.active_steps = self.active_steps.to(device)
        self.max_success_rate = self.max_success_rate.to(device)
        self.transport_error_sum = self.transport_error_sum.to(device)
        self.transport_velocity_sum = self.transport_velocity_sum.to(device)
        self.contact_distance_sum = self.contact_distance_sum.to(device)
        self.collision_rate_sum = self.collision_rate_sum.to(device)
        self.engagement_rate_sum = self.engagement_rate_sum.to(device)
        self.push_alignment_sum = self.push_alignment_sum.to(device)

        self.max_success_rate = torch.max(self.max_success_rate, metrics["success_rate"])

        # 1. Flag newly completed environments
        newly_completed = (metrics["success_rate"] == 1.0) & (~self.is_completed)
        self.is_completed = self.is_completed | (metrics["success_rate"] == 1.0)
        
        # 2. Log exact completion time for the newly finished games
        self.completion_times[newly_completed] = self.total_steps
        
        # 3. Active Masking: Accumulate metrics ONLY for games that are still running
        active_mask = ~self.is_completed
        
        self.active_steps[active_mask] += 1
        self.transport_error_sum[active_mask] += metrics["transport_error"][active_mask]
        self.transport_velocity_sum[active_mask] += metrics["transport_velocity"][active_mask]
        self.contact_distance_sum[active_mask] += metrics["contact_distance"][active_mask]
        self.collision_rate_sum[active_mask] += metrics["collision_rate"][active_mask]
        self.engagement_rate_sum[active_mask] += metrics["engagement_rate"][active_mask]
        self.push_alignment_sum[active_mask] += metrics["push_alignment"][active_mask]
            
        self.total_steps += 1

    def _get_per_game_stats(self, sum_tensor):
        # Average the accumulated metric over the exact number of steps that game was active
        valid_steps = torch.clamp(self.active_steps, min=1.0)
        per_game_avg = sum_tensor / valid_steps
        
        mean_val = per_game_avg.mean().item()
        
        # Prevent NaN standard deviations if running a single game test
        sd_val = per_game_avg.std(unbiased=False).item() if self.num_games > 1 else 0.0
        
        return round(mean_val, 4), round(sd_val, 4)

    def get_summary(self):
        te_m, te_sd = self._get_per_game_stats(self.transport_error_sum)
        tv_m, tv_sd = self._get_per_game_stats(self.transport_velocity_sum)
        cd_m, cd_sd = self._get_per_game_stats(self.contact_distance_sum)
        cr_m, cr_sd = self._get_per_game_stats(self.collision_rate_sum)
        eng_m, eng_sd = self._get_per_game_stats(self.engagement_rate_sum)
        align_m, align_sd = self._get_per_game_stats(self.push_alignment_sum)

        # Calculate final Success Rate
        success_rate = self.max_success_rate.mean().item()
        
        # Time-to-Completion Calculation across games
        ttc = self.completion_times.clone()
        ttc[torch.isnan(ttc)] = self.total_steps
        ttc_m = round(float(ttc.mean().item()), 2)
        ttc_sd = round(float(ttc.std(unbiased=False).item()), 2) if self.num_games > 1 else 0.0

        return {
            "Overall_Success_Rate": round(success_rate, 4),
            "Mean_Time_To_Completion": ttc_m,
            "SD_Time_To_Completion": ttc_sd,
            "Active_Transport_Error_Mean": te_m, "Active_Transport_Error_SD": te_sd,
            "Active_Transport_Velocity_Mean": tv_m, "Active_Transport_Velocity_SD": tv_sd,
            "Active_Contact_Distance_Mean": cd_m, "Active_Contact_Distance_SD": cd_sd,
            "Active_Collision_Rate_Mean": cr_m, "Active_Collision_Rate_SD": cr_sd,
            "Active_Engagement_Rate_Mean": eng_m, "Active_Engagement_Rate_SD": eng_sd,
            "Active_Push_Alignment_Mean": align_m, "Active_Push_Alignment_SD": align_sd,
        }

def parse_harvest_args():
    parser = argparse.ArgumentParser()
    
    # --- Mode Configuration ---
    parser.add_argument("--mode", type=str, choices=["harvest", "inference"], default="inference")
    parser.add_argument("--prefix", type=str, default="")
    
    # --- Inference Arguments ---
    parser.add_argument("--n-test-array", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--num-games-per-test", type=int, default=3, 
                        help="Number of parallel environments to run per N_test for metric aggregation.")
    parser.add_argument("--csv-output", type=str, default="transport_metrics.csv")
    
    # --- Standard Harvest Arguments ---
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-landmarks", type=int, default=20, help="Agent count for harvest mode")
    parser.add_argument("--num-packages", type=int, default=2, help="Number of packages to transport")
    parser.add_argument("--n-max", type=int, default=8, help="Must match GNN context window size during training")
    parser.add_argument("--num-trajectories", type=int, default=500000)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="expert_data_transport")
    parser.add_argument("--video-interval", type=int, default=2000)
    parser.add_argument("--max-cycles", type=int, default=2000)
    
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    args.env_id = "transport"
    args.capture_video = False 
    
    # Determine actual num_envs based on mode
    if args.mode == "inference":
        # Will be updated dynamically per test
        args.num_envs = args.num_landmarks * args.num_games_per_test
    else:
        args.num_envs = args.num_landmarks # Default to 1 game for harvesting
        
    return args

def load_oracle_model(args, envs, device):
    """Initializes the architecture and loads strict weights."""
    state_dim = envs.num_agents * np.array(envs.single_observation_space.shape).prod()

    oracle = PointNetAgent(
        envs.single_action_space, 
        envs.single_observation_space.shape, 
        num_agents=envs.num_agents, 
        state_dim=state_dim, 
        n_max=args.n_max
    ).to(device)

    # oracle = GraphAgent(
    #     envs=envs, 
    #     n_max=args.n_max, 
    #     num_agents=envs.num_agents
    # ).to(device)

    print(f"Loading Oracle weights from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    
    # Strip out the centralized critic parameters to avoid dimension mismatch on N scaling
    keys_to_remove = [k for k in state_dict.keys() if "critic" in k or "embedding" in k]
    for k in keys_to_remove:
        if k in state_dict:
            del state_dict[k]
            
    oracle.load_state_dict(state_dict, strict=False)
    oracle.eval()
    return oracle

def get_action(oracle, obs):
    """Performs deterministic forward pass across environments.[cite: 1]"""
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
    
    # Slight exploration noise to prevent deadlock configurations[cite: 1]
    noise = torch.randn_like(deterministic_action) * 0.025
    clipped_action = torch.clamp(deterministic_action + noise, -1.0, 1.0)
    
    return clipped_action

def run_inference(args):
    """Runs a multi-environment batch for each N in --n-test-array."""
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    all_metrics = []
    
    prefix_str = f"{args.prefix}_" if args.prefix else ""
    video_dir = os.path.join(args.output_dir, f"{prefix_str}inference_videos")
    os.makedirs(video_dir, exist_ok=True)
    
    csv_dirname = os.path.dirname(args.csv_output)
    csv_basename = os.path.basename(args.csv_output)
    final_csv_path = os.path.join(csv_dirname, f"{prefix_str}{csv_basename}") if csv_dirname else f"{prefix_str}{csv_basename}"
    
    print(f"\n--- STARTING INFERENCE BATCH SWEEP ---")
    print(f"Oracle Model: {args.model_path}")
    print(f"Configurations to evaluate: {args.n_test_array}")
    print(f"Games per configuration: {args.num_games_per_test}")
    
    for n_test in args.n_test_array:
        print(f"\nEvaluating N = {n_test} (x{args.num_games_per_test} environments)...")
        
        args.num_landmarks = n_test
        args.num_envs = n_test * args.num_games_per_test
        
        envs = VMASVectorizedEnv(args, args.seed, run_name=f"{prefix_str}infer_run_{n_test}", update_step=0)
        tracker = TransportMetricTracker(envs.num_games, envs.num_agents)
        oracle = load_oracle_model(args, envs, device)
        
        reset_data = envs.reset(seed=args.seed)
        if isinstance(reset_data, tuple):
            obs = reset_data[0].clone().to(device)
        else:
            obs = reset_data.clone().to(device)
            
        video_frames = []
            
        with torch.no_grad():
            for step in tqdm(range(args.max_cycles), desc=f"Episode Progress (N={n_test})"):
                action = get_action(oracle, obs)
                
                # Update physical metrics using the environment's world state
                tracker.update(envs.env.world)
                
                # Render only the first environment out of the batch for video profiling
                current_frames = []
                for i in range(args.num_games_per_test):
                    frame = envs.env.render(mode="rgb_array", env_index=i, agent_index_focus=None)
                    if isinstance(frame, list): 
                        frame = frame[0]
                    current_frames.append(frame)

                n = len(current_frames)
                cols = math.ceil(math.sqrt(n))
                rows = math.ceil(n / cols)
                H, W, C = current_frames[0].shape
                blank = np.zeros((H, W, C), dtype=np.uint8)
                
                while len(current_frames) < rows * cols:
                    current_frames.append(blank)
                    
                grid = np.vstack([np.hstack(current_frames[i*cols:(i+1)*cols]) for i in range(rows)])
                video_frames.append(grid)
                
                step_data = envs.step(action)
                obs = step_data[0].clone().to(device)

        video_path = os.path.join(video_dir, f"{prefix_str}inference_N{n_test}_P{args.num_packages}.mp4")
        imageio.mimsave(video_path, video_frames, fps=15)

        metrics = tracker.get_summary()
        metrics = {"N_test": n_test, "Packages": args.num_packages, "Games_Sampled": args.num_games_per_test, **metrics}
        all_metrics.append(metrics)
        
        for k, v in metrics.items():
            print(f"  {k}: {v}")
            
        envs.close()

    # Aggregate and Save
    if all_metrics:
        os.makedirs(os.path.dirname(os.path.abspath(final_csv_path)), exist_ok=True)
        keys = all_metrics[0].keys()
        
        with open(final_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_metrics)
            
        print(f"\n[Success] Batch inference metrics saved to: {final_csv_path}")

def harvest_imitation_data(args):
    """Runs long-term data collection up to --num-trajectories."""
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    video_dir = os.path.join(args.output_dir, "expert_videos")
    os.makedirs(video_dir, exist_ok=True)

    video_frames = []
    is_recording = False
    
    envs = VMASVectorizedEnv(args, args.seed, run_name="harvest_run", update_step=0)
    video_tracker = TransportMetricTracker(envs.num_games, envs.num_agents)
    oracle = load_oracle_model(args, envs, device)
    
    expert_obs = []
    expert_actions = []
    chunk_idx = 0
    
    reset_data = envs.reset(seed=args.seed)
    if isinstance(reset_data, tuple):
        obs = reset_data[0].clone().to(device)
    else:
        obs = reset_data.clone().to(device)
    
    print("Starting data collection...")
    with torch.no_grad():
        for step in tqdm(range(args.num_trajectories)):
            if step % args.video_interval == 0:
                is_recording = True
                video_frames = []
                video_tracker.reset()

            clipped_action = get_action(oracle, obs)

            if is_recording:
                video_tracker.update(envs.env.world)
                frame = envs.env.render(mode="rgb_array", env_index=0, agent_index_focus=None)
                if isinstance(frame, list):
                    frame = frame[0]
                video_frames.append(frame)
                
                if len(video_frames) >= args.max_cycles:
                    video_path = os.path.join(video_dir, f"expert_step_{step}.mp4")
                    metrics_path = os.path.join(video_dir, f"expert_step_{step}_metrics.json")
                    imageio.mimsave(video_path, video_frames, fps=15)

                    with open(metrics_path, "w") as f:
                        json.dump(video_tracker.get_summary(), f, indent=2)
                        
                    video_frames = []
                    is_recording = False
            
            expert_obs.append(obs.cpu().numpy())
            expert_actions.append(clipped_action.cpu().numpy())
            
            step_data = envs.step(clipped_action)
            obs = step_data[0].clone().to(device)
            
            if len(expert_obs) >= args.chunk_size:
                np.save(os.path.join(args.output_dir, f"obs_N{args.num_landmarks}_part{chunk_idx}.npy"), np.vstack(expert_obs))
                np.save(os.path.join(args.output_dir, f"actions_N{args.num_landmarks}_part{chunk_idx}.npy"), np.vstack(expert_actions))
                
                expert_obs.clear()
                expert_actions.clear()
                gc.collect()
                chunk_idx += 1
                
    if len(expert_obs) > 0:
        np.save(os.path.join(args.output_dir, f"obs_N{args.num_landmarks}_part{chunk_idx}.npy"), np.vstack(expert_obs))
        np.save(os.path.join(args.output_dir, f"actions_N{args.num_landmarks}_part{chunk_idx}.npy"), np.vstack(expert_actions))
        
    envs.close()

if __name__ == "__main__":
    args = parse_harvest_args()
    
    if args.mode == "harvest":
        harvest_imitation_data(args)
    elif args.mode == "inference":
        run_inference(args)
import argparse
import os
import gc
import csv
import torch
import numpy as np
import imageio
from tqdm import tqdm
import json

# Import the architecture and environment wrapper directly from your flocking training script
# from ppo_vmas_flocking_gnn import GraphAgent, VMASVectorizedEnv
from ppo_vmas_flocking_mappo import Agent, PointNetAgent, VMASVectorizedEnv

class FlockingMetricTracker:
    def __init__(self, num_games, num_agents, agent_radius=0.1, desired_spacing=0.4, n_max=5, comm_radius=2.0):
        self.num_games = num_games
        self.num_agents = num_agents
        self.agent_radius = agent_radius
        self.desired_spacing = desired_spacing
        self.n_max = n_max
        self.comm_radius = comm_radius
        
        self.rdf_max_r = 2.0
        self.rdf_bins = 40
        self.rdf_bin_edges = torch.linspace(0, self.rdf_max_r, self.rdf_bins + 1)
        
        self.reset()

    def reset(self):
        self.total_steps = 0
        self.prev_actions = None
        self.prev_com = None
        self.initial_target_dist = None
        self.com_distance_traveled = None
        
        # Keep histories as lists of tensors to prevent CPU blocking
        self.polarization_history = []
        self.cohesion_history = []
        self.tracking_error_history = []
        self.collision_rate_history = []
        self.speed_variance_history = []
        self.action_jitter_history = []
        self.isolation_rate_history = []
        
        self.rdf_histogram = torch.zeros(self.rdf_bins)

    def update(self, raw_obs_flat, actions_flat):
        device = raw_obs_flat.device
        obs_dim = raw_obs_flat.shape[-1]
        
        raw_obs = raw_obs_flat.view(self.num_games, self.num_agents, obs_dim)
        actions = actions_flat.view(self.num_games, self.num_agents, -1)

        pos = raw_obs[:, :, 0:2]
        vel = raw_obs[:, :, 2:4]
        target_rel = raw_obs[:, 0:1, 4:6]
        target_abs = pos[:, 0:1, :] + target_rel
        com = pos.mean(dim=1, keepdim=True)

        if self.prev_com is None:
            self.prev_com = com.clone()
            self.prev_actions = actions.clone()
            self.initial_target_dist = torch.norm(com - target_abs, dim=-1).squeeze(1)
            self.com_distance_traveled = torch.zeros(self.num_games, device=device)
            self.rdf_bin_edges = self.rdf_bin_edges.to(device)
            self.rdf_histogram = self.rdf_histogram.to(device)

        speeds = torch.norm(vel, dim=-1)
        headings = vel / (speeds.unsqueeze(-1) + 1e-8)
        mean_headings = headings.sum(dim=1) / self.num_agents
        
        polarization = torch.norm(mean_headings, dim=-1)
        cohesion_spread = torch.norm(pos - com, dim=-1).mean(dim=1)
        tracking_error = torch.norm(com - target_abs, dim=-1).squeeze(1)

        pos_expanded_1 = pos.unsqueeze(2)
        pos_expanded_2 = pos.unsqueeze(1)
        pairwise_dist = torch.norm(pos_expanded_1 - pos_expanded_2, dim=-1)
        
        eye_mask = torch.eye(self.num_agents, device=device).unsqueeze(0).bool()
        pairwise_dist.masked_fill_(eye_mask, float('inf'))

        valid_dists = pairwise_dist[pairwise_dist < self.rdf_max_r]
        if len(valid_dists) > 0:
            hist = torch.histc(valid_dists, bins=self.rdf_bins, min=0.0, max=self.rdf_max_r)
            self.rdf_histogram += hist

        is_colliding = (pairwise_dist < (self.agent_radius * 2.0)).float()
        collision_rate = is_colliding.sum(dim=-1).mean(dim=1)

        neighbors_in_range = (pairwise_dist < self.comm_radius).sum(dim=-1)
        is_isolated = (neighbors_in_range < self.n_max).float()
        isolation_rate = is_isolated.mean(dim=1)

        speed_variance = speeds.var(dim=1)
        step_jitter = torch.norm(actions - self.prev_actions, dim=-1)
        action_jitter = step_jitter.mean(dim=1)
        step_com_dist = torch.norm(com - self.prev_com, dim=-1).squeeze(1)
        self.com_distance_traveled += step_com_dist

        self.prev_com = com.clone()
        self.prev_actions = actions.clone()

        # Append pure tensors (NO .item() calls here)
        self.polarization_history.append(polarization.mean())
        self.cohesion_history.append(cohesion_spread.mean())
        self.tracking_error_history.append(tracking_error.mean())
        self.collision_rate_history.append(collision_rate.mean())
        self.speed_variance_history.append(speed_variance.mean())
        self.action_jitter_history.append(action_jitter.mean())
        self.isolation_rate_history.append(isolation_rate.mean())
        
        self.total_steps += 1

    def _get_stats(self, tensor_list):
        if not tensor_list: return 0.0, 0.0
        # Sync to CPU once at the very end
        arr = torch.stack(tensor_list).cpu().numpy()
        return round(float(np.mean(arr)), 4), round(float(np.std(arr)), 4)

    def get_summary(self):
        valid_tortuosity_mask = self.initial_target_dist > 0.1
        if valid_tortuosity_mask.any():
            tortuosity = (self.com_distance_traveled[valid_tortuosity_mask] / self.initial_target_dist[valid_tortuosity_mask]).mean().item()
        else:
            tortuosity = 1.0

        global_density = self.num_agents / (np.pi * (self.comm_radius ** 2))
        bin_areas = np.pi * (self.rdf_bin_edges[1:]**2 - self.rdf_bin_edges[:-1]**2)
        bin_areas = bin_areas.to(self.rdf_histogram.device)
        
        normalization_factor = bin_areas * global_density * self.total_steps * self.num_games * self.num_agents
        g_r = self.rdf_histogram / (normalization_factor + 1e-8)
        
        target_bin_idx = torch.bucketize(torch.tensor(self.desired_spacing), self.rdf_bin_edges.cpu()) - 1
        target_bin_idx = torch.clamp(target_bin_idx, 0, self.rdf_bins - 1).item()
        lattice_structure_score = g_r[target_bin_idx].item()

        # Calculate Mean and SD for all series
        pol_m, pol_sd = self._get_stats(self.polarization_history)
        speed_m, speed_sd = self._get_stats(self.speed_variance_history)
        coh_m, coh_sd = self._get_stats(self.cohesion_history)
        trk_m, trk_sd = self._get_stats(self.tracking_error_history)
        col_m, col_sd = self._get_stats(self.collision_rate_history)
        iso_m, iso_sd = self._get_stats(self.isolation_rate_history)
        jit_m, jit_sd = self._get_stats(self.action_jitter_history)

        return {
            "Mean_Polarization": pol_m, "SD_Polarization": pol_sd,
            "Mean_Speed_Variance": speed_m, "SD_Speed_Variance": speed_sd,
            "Mean_Cohesion_Spread": coh_m, "SD_Cohesion_Spread": coh_sd,
            "Mean_Tracking_Error": trk_m, "SD_Tracking_Error": trk_sd,
            "CoM_Tortuosity": round(float(tortuosity), 4),
            "Mean_Collision_Rate": col_m, "SD_Collision_Rate": col_sd,
            "Mean_Isolation_Rate": iso_m, "SD_Isolation_Rate": iso_sd,
            "Mean_Action_Jitter": jit_m, "SD_Action_Jitter": jit_sd,
            f"g(r)_Peak_at_{self.desired_spacing}m": round(float(lattice_structure_score), 4)
        }

def parse_harvest_args():
    parser = argparse.ArgumentParser()
    
    # --- Mode Configuration ---[cite: 1]
    parser.add_argument("--mode", type=str, choices=["harvest", "inference"], default="harvest")
    parser.add_argument("--prefix", type=str, default="")
    
    # --- Inference Arguments ---[cite: 1]
    parser.add_argument("--n-test-array", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--num-games-per-test", type=int, default=3, 
                        help="Number of parallel environments to run per N_test for metric aggregation.")
    parser.add_argument("--csv-output", type=str, default="flocking_metrics.csv")
    
    # --- Standard Harvest Arguments ---[cite: 1]
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-landmarks", type=int, default=10, help="N config for harvest mode")
    parser.add_argument("--n-max", type=int, default=5, help="Must match GNN context window size during training")
    parser.add_argument("--num-trajectories", type=int, default=500000)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="expert_data_flocking")
    parser.add_argument("--video-interval", type=int, default=10000)
    parser.add_argument("--max-cycles", type=int, default=1000)
    
    args = parser.parse_args()
    
    args.cuda = torch.cuda.is_available()
    args.env_id = "flocking"
    args.capture_video = False 
    
    # Determine actual num_envs based on mode
    if args.mode == "inference":
        # Will be updated dynamically per test
        args.num_envs = args.num_landmarks * args.num_games_per_test
    else:
        args.num_envs = args.num_landmarks # Default to 1 game for harvesting
        
    return args

def load_oracle_model(args, envs, device):
    """Initializes the architecture and loads strict weights.[cite: 1]"""
    state_dim = envs.num_agents * np.array(envs.single_observation_space.shape).prod()

    # oracle = Agent(
    #         envs.single_action_space, 
    #         envs.single_observation_space.shape, 
    #         num_agents = envs.num_agents, 
    #         state_dim=state_dim, 
    #         n_max=args.n_max
    #     ).to(device)

    oracle = PointNetAgent(
        envs.single_action_space, 
        envs.single_observation_space.shape, 
        num_agents = envs.num_agents, 
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
    """Runs a multi-environment batch for each N in --n-test-array.[cite: 1]"""
    device = torch.device("cuda" if args.cuda else "cpu")
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
        tracker = FlockingMetricTracker(envs.num_games, envs.num_agents)
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
                
                # Render only the first environment out of the batch for video profiling
                frame = envs.env.render(mode="rgb_array", env_index=0, agent_index_focus=None)
                if isinstance(frame, list):
                    frame = frame[0]
                video_frames.append(frame)
                
                step_data = envs.step(action)
                obs = step_data[0].clone().to(device)
                if len(step_data) >= 4 and isinstance(step_data[-1], dict) and "raw_obs" in step_data[-1]:
                    raw_obs = step_data[-1]["raw_obs"].clone().to(device)

        video_path = os.path.join(video_dir, f"{prefix_str}inference_N{n_test}.mp4")
        imageio.mimsave(video_path, video_frames, fps=15)

        metrics = tracker.get_summary()
        metrics = {"N_test": n_test, "Games_Sampled": args.num_games_per_test, **metrics}
        all_metrics.append(metrics)
        
        for k, v in metrics.items():
            print(f"  {k}: {v}")
            
        envs.close()

    # Aggregate and Save[cite: 1]
    if all_metrics:
        os.makedirs(os.path.dirname(os.path.abspath(final_csv_path)), exist_ok=True)
        keys = all_metrics[0].keys()
        
        with open(final_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_metrics)
            
        print(f"\n[Success] Batch inference metrics saved to: {final_csv_path}")

def harvest_imitation_data(args):
    """Runs long-term data collection up to --num-trajectories.[cite: 1]"""
    device = torch.device("cuda" if args.cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    video_dir = os.path.join(args.output_dir, "expert_videos")
    os.makedirs(video_dir, exist_ok=True)

    video_frames = []
    is_recording = False
    
    envs = VMASVectorizedEnv(args, args.seed, run_name="harvest_run", update_step=0)
    video_tracker = FlockingMetricTracker(envs.num_games, envs.num_agents)
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

                    with open(metrics_path, "w") as f:
                        json.dump(video_tracker.get_summary(), f, indent=2)
                        
                    video_frames = []
                    is_recording = False
            
            expert_obs.append(obs.cpu().numpy())
            expert_actions.append(clipped_action.cpu().numpy())
            
            step_data = envs.step(clipped_action)
            obs = step_data[0].clone().to(device)
            if len(step_data) >= 4 and isinstance(step_data[-1], dict) and "raw_obs" in step_data[-1]:
                raw_obs = step_data[-1]["raw_obs"].clone().to(device)
            
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
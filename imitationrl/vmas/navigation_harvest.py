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

# Import the architecture and environment wrapper directly from your training script
# from ppo_vmas_navigation_gnn import GraphAgent, VMASVectorizedEnv
# from ppo_vmas_navigation_mappo import Agent, TransformerAgent, PointNetAgent, VMASVectorizedEnv
from ppo_vmas_navigation_radius import MAPPOAgent, TransformerAgent, PointNetAgent, GraphAgent, VMASVectorizedEnv

class BehavioralMetricTracker:
    def __init__(self, num_games, num_agents, agent_radius=0.1, contact_threshold=0.20, goal_tolerance=0.25):
        self.num_games = num_games
        self.num_agents = num_agents
        self.agent_radius = agent_radius
        self.contact_threshold = contact_threshold
        self.goal_tolerance = goal_tolerance
        self.reset()

    def reset(self):
        self.total_steps = 0
        
        # Vectorized accumulators (Shape: [num_games])
        self.active_agent_steps = torch.zeros(self.num_games)
        self.settled_agent_steps = torch.zeros(self.num_games)
        
        self.cooperative_yields = torch.zeros(self.num_games)
        self.forced_displacements = torch.zeros(self.num_games)
        
        self.deadlock_events = torch.zeros(self.num_games)
        self.collision_events = torch.zeros(self.num_games)
        
        self.min_clearances_sum = torch.zeros(self.num_games)
        
        self.free_speeds_sum = torch.zeros(self.num_games)
        self.free_speeds_count = torch.zeros(self.num_games)
        self.congested_speeds_sum = torch.zeros(self.num_games)
        self.congested_speeds_count = torch.zeros(self.num_games)
        
        self.active_energy_sum = torch.zeros(self.num_games)
        self.active_jitter_sum = torch.zeros(self.num_games)
        
        self.settled_energy_sum = torch.zeros(self.num_games)
        self.settled_jitter_sum = torch.zeros(self.num_games)
        self.max_settled_energy = torch.zeros(self.num_games)
        
        # Tracking states
        self.currently_at_goal = None
        self.final_goal_distances = None
        self.done_mask = None
        self.convergence_steps = None
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
        device = pos.device
        
        goal_dists = torch.norm(to_goal, dim=-1) # [B, N]
        self.final_goal_distances = goal_dists.clone()
        
        if self.start_positions is None:
            self.start_positions = pos.clone()
            self.prev_positions = pos.clone()
            self.initial_goal_distances = goal_dists.clone()
            self.distance_traveled = torch.zeros((self.num_games, self.num_agents), device=device)
            self.prev_actions = actions.clone()
            
            self.done_mask = torch.zeros((self.num_games, self.num_agents), dtype=torch.bool, device=device)
            self.convergence_steps = torch.full((self.num_games, self.num_agents), float('inf'), device=device)
            self.currently_at_goal = goal_dists < self.goal_tolerance
            
            # Move accumulators to device
            self.active_agent_steps = self.active_agent_steps.to(device)
            self.settled_agent_steps = self.settled_agent_steps.to(device)
            self.cooperative_yields = self.cooperative_yields.to(device)
            self.forced_displacements = self.forced_displacements.to(device)
            self.deadlock_events = self.deadlock_events.to(device)
            self.collision_events = self.collision_events.to(device)
            self.min_clearances_sum = self.min_clearances_sum.to(device)
            self.free_speeds_sum = self.free_speeds_sum.to(device)
            self.free_speeds_count = self.free_speeds_count.to(device)
            self.congested_speeds_sum = self.congested_speeds_sum.to(device)
            self.congested_speeds_count = self.congested_speeds_count.to(device)
            self.active_energy_sum = self.active_energy_sum.to(device)
            self.active_jitter_sum = self.active_jitter_sum.to(device)
            self.settled_energy_sum = self.settled_energy_sum.to(device)
            self.settled_jitter_sum = self.settled_jitter_sum.to(device)
            self.max_settled_energy = self.max_settled_energy.to(device)
            return

        self.total_steps += 1
        
        # --- GLOBAL DISTANCES ---
        pos_i = pos.unsqueeze(2)
        pos_j = pos.unsqueeze(1)
        dist_matrix = torch.norm(pos_i - pos_j, dim=-1)
        mask = torch.eye(self.num_agents, device=device).bool().unsqueeze(0)
        dist_matrix.masked_fill_(mask, float('inf'))
        closest_dist, _ = dist_matrix.min(dim=-1) # [B, N]

        # --- ADVANCED YIELD TRACKING ---
        at_goal_now = goal_dists < self.goal_tolerance
        just_departed = self.currently_at_goal & (~at_goal_now)
        
        step_jitters = torch.norm(actions - self.prev_actions, dim=-1)
        is_collision_free = closest_dist >= self.contact_threshold
        is_intentional = step_jitters < 0.5 
        
        true_yields = just_departed & is_collision_free & is_intentional
        forced_bumps = just_departed & (~(is_collision_free & is_intentional))
        
        self.cooperative_yields += true_yields.sum(dim=1)
        self.forced_displacements += forced_bumps.sum(dim=1)
        
        self.currently_at_goal = at_goal_now.clone()
        
        # --- CONVERGENCE & MASKS ---
        just_finished = (goal_dists < self.goal_tolerance) & (~self.done_mask)
        self.convergence_steps[just_finished] = self.total_steps
        self.done_mask = self.done_mask | (goal_dists < self.goal_tolerance)
        
        active_mask = ~self.done_mask
        settled_mask = self.done_mask
        
        self.active_agent_steps += active_mask.sum(dim=1)
        self.settled_agent_steps += settled_mask.sum(dim=1)
        
        action_norms = torch.norm(actions, dim=-1)

        # --- SETTLED PHASE METRICS ---
        current_max_energy = torch.where(settled_mask, action_norms, torch.zeros_like(action_norms)).max(dim=1)[0]
        self.max_settled_energy = torch.max(self.max_settled_energy, current_max_energy)
        
        self.settled_energy_sum += torch.where(settled_mask, action_norms, torch.zeros_like(action_norms)).sum(dim=1)
        self.settled_jitter_sum += torch.where(settled_mask, step_jitters, torch.zeros_like(step_jitters)).sum(dim=1)

        # --- ACTIVE PHASE METRICS ---
        self.min_clearances_sum += torch.where(active_mask, closest_dist, torch.zeros_like(closest_dist)).sum(dim=1)
        self.collision_events += (active_mask & (closest_dist < self.contact_threshold)).sum(dim=1)
        
        speeds = torch.norm(vel, dim=-1)
        active_congested = active_mask & (closest_dist < (self.agent_radius * 4.0))
        active_free = active_mask & (~active_congested)
        
        self.congested_speeds_sum += torch.where(active_congested, speeds, torch.zeros_like(speeds)).sum(dim=1)
        self.congested_speeds_count += active_congested.sum(dim=1)
        
        self.free_speeds_sum += torch.where(active_free, speeds, torch.zeros_like(speeds)).sum(dim=1)
        self.free_speeds_count += active_free.sum(dim=1)
        
        is_deadlocked = active_mask & (speeds < 0.05) & (goal_dists > self.goal_tolerance)
        self.deadlock_events += is_deadlocked.sum(dim=1)
        
        self.active_energy_sum += torch.where(active_mask, action_norms, torch.zeros_like(action_norms)).sum(dim=1)
        self.active_jitter_sum += torch.where(active_mask, step_jitters, torch.zeros_like(step_jitters)).sum(dim=1)
        
        step_distances = torch.norm(pos - self.prev_positions, dim=-1)
        self.distance_traveled += torch.where(active_mask, step_distances, torch.zeros_like(step_distances))
        
        self.prev_positions = pos.clone()
        self.prev_actions = actions.clone()

    def _get_stats(self, tensor_vals):
        mean_val = tensor_vals.mean().item()
        sd_val = tensor_vals.std(unbiased=False).item() if self.num_games > 1 else 0.0
        return round(mean_val, 3), round(sd_val, 3)

    def get_summary(self):
        retained = (self.final_goal_distances < self.goal_tolerance).float()
        success_rate_per_game = retained.mean(dim=1) * 100.0
        s_mean, s_sd = self._get_stats(success_rate_per_game)
        
        converged_mask = self.convergence_steps < float('inf')
        t_conv_per_game = torch.zeros(self.num_games, device=self.convergence_steps.device)
        for i in range(self.num_games):
            valid_t = self.convergence_steps[i][converged_mask[i]]
            t_conv_per_game[i] = valid_t.mean() if len(valid_t) > 0 else self.total_steps
        t_mean, t_sd = self._get_stats(t_conv_per_game)
        
        yields_c_mean, yields_c_sd = self._get_stats(self.cooperative_yields)
        yields_f_mean, yields_f_sd = self._get_stats(self.forced_displacements)
        
        safe_active_steps = torch.clamp(self.active_agent_steps, min=1.0)
        safe_settled_steps = torch.clamp(self.settled_agent_steps, min=1.0)
        safe_free_count = torch.clamp(self.free_speeds_count, min=1.0)
        safe_cong_count = torch.clamp(self.congested_speeds_count, min=1.0)
        
        f_rate_per_game = (self.deadlock_events / safe_active_steps) * 100.0
        f_mean, f_sd = self._get_stats(f_rate_per_game)
        
        c_rate_per_game = (self.collision_events / safe_active_steps) * 100.0
        c_mean, c_sd = self._get_stats(c_rate_per_game)
        
        d_min_per_game = self.min_clearances_sum / safe_active_steps
        d_min_mean, d_min_sd = self._get_stats(d_min_per_game)
        
        v_free_per_game = self.free_speeds_sum / safe_free_count
        v_cong_per_game = self.congested_speeds_sum / safe_cong_count
        v_deg_per_game = v_cong_per_game / torch.clamp(v_free_per_game, min=1e-5)
        v_deg_mean, v_deg_sd = self._get_stats(v_deg_per_game)
        
        valid_goals = self.initial_goal_distances > 0.1
        valid_tortuosity = valid_goals & converged_mask
        tau_per_game = torch.ones(self.num_games, device=self.distance_traveled.device)
        for i in range(self.num_games):
            vt = valid_tortuosity[i]
            if vt.any():
                tau_per_game[i] = (self.distance_traveled[i][vt] / self.initial_goal_distances[i][vt]).mean()
        tau_mean, tau_sd = self._get_stats(tau_per_game)
        
        e_act_per_game = self.active_energy_sum / safe_active_steps
        e_act_mean, e_act_sd = self._get_stats(e_act_per_game)
        
        j_act_per_game = self.active_jitter_sum / safe_active_steps
        j_act_mean, j_act_sd = self._get_stats(j_act_per_game)
        
        e_set_per_game = self.settled_energy_sum / safe_settled_steps
        e_set_mean, e_set_sd = self._get_stats(e_set_per_game)
        
        j_set_per_game = self.settled_jitter_sum / safe_settled_steps
        j_set_mean, j_set_sd = self._get_stats(j_set_per_game)
        
        e_max_mean, e_max_sd = self._get_stats(self.max_settled_energy)

        return {
            "S_rate_Mean": s_mean, "S_rate_SD": s_sd,
            "T_conv_Mean": t_mean, "T_conv_SD": t_sd,
            "Yields_Cooperative_Mean": yields_c_mean, "Yields_Cooperative_SD": yields_c_sd,
            "Yields_Forced_Mean": yields_f_mean, "Yields_Forced_SD": yields_f_sd,
            "F_rate_Mean": f_mean, "F_rate_SD": f_sd,
            "C_rate_Mean": c_mean, "C_rate_SD": c_sd,
            "d_min_Mean": d_min_mean, "d_min_SD": d_min_sd,
            "V_deg_Mean": v_deg_mean, "V_deg_SD": v_deg_sd,
            "Tau_Mean": tau_mean, "Tau_SD": tau_sd,
            "E_active_Mean": e_act_mean, "E_active_SD": e_act_sd,
            "J_active_Mean": j_act_mean, "J_active_SD": j_act_sd,
            "E_settled_Mean": e_set_mean, "E_settled_SD": e_set_sd,
            "E_settled_max_Mean": e_max_mean, "E_settled_max_SD": e_max_sd,
            "J_settled_Mean": j_set_mean, "J_settled_SD": j_set_sd
        }

def parse_harvest_args():
    parser = argparse.ArgumentParser()
    
    # --- Mode Configuration ---
    parser.add_argument("--mode", type=str, choices=["harvest", "inference"], default="harvest")
    parser.add_argument("--prefix", type=str, default="")
    
    # --- Inference Arguments ---
    parser.add_argument("--n-test-array", type=int, nargs="+", default=[5, 7, 10])
    # ADDED: Match transport scaling
    parser.add_argument("--num-games-per-test", type=int, default=3, 
                        help="Number of parallel environments to run per N_test for metric aggregation.")
    parser.add_argument("--csv-output", type=str, default="inference_metrics.csv")
    
    # --- Standard Harvest Arguments ---
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-landmarks", type=int, default=7)
    parser.add_argument("--n-max", type=int, default=16)
    parser.add_argument("--num-trajectories", type=int, default=500000)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./expert_data")
    parser.add_argument("--video-interval", type=int, default=10000)
    parser.add_argument("--max-cycles", type=int, default=350)
    
    args = parser.parse_args()
    
    args.cuda = torch.cuda.is_available()
    args.env_id = "navigation"
    args.capture_video = False 
    
    # Determine actual num_envs based on mode
    if args.mode == "inference":
        args.num_envs = args.num_landmarks * args.num_games_per_test
    else:
        args.num_envs = args.num_landmarks 
        
    return args

def load_oracle_model(args, envs, device):
    state_dim = envs.num_agents * np.array(envs.single_observation_space.shape).prod()
    
    n_max = args.n_max * 2
    # oracle = MAPPOAgent(
    #     envs.single_action_space, 
    #     envs.single_observation_space.shape, 
    #     envs.num_agents, 
    #     state_dim=state_dim, 
    #     n_max=n_max
    # ).to(device)

    # oracle = TransformerAgent(
    #     envs.single_action_space, 
    #     envs.single_observation_space.shape, 
    #     envs.num_agents, 
    #     state_dim=state_dim, 
    #     n_max=n_max
    # ).to(device)

    oracle = PointNetAgent(
        envs.single_action_space, 
        envs.single_observation_space.shape, 
        num_agents = envs.num_agents, 
        state_dim=state_dim, 
        n_max=n_max
    ).to(device)
    
    # oracle = GraphAgent(
    #     envs=envs, 
    #     n_max=n_max, 
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
    # oracle.train()
    return oracle

def get_action(oracle, obs):
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
        backbone_outputs = oracle._forward_actor_backbone(obs_norm)
        actor_features = oracle.rho(backbone_outputs)
    else:
        actor_features = oracle.actor(obs_norm)
        
    deterministic_action = oracle.actor_mean(actor_features)
    noise = torch.randn_like(deterministic_action) * 0.025
    clipped_action = torch.clamp(deterministic_action + noise, -1.0, 1.0)
    
    return clipped_action

def run_inference(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    all_metrics = []
    
    prefix_str = f"{args.prefix}_" if args.prefix else ""
    
    video_dir = os.path.join(args.output_dir, f"{prefix_str}inference_videos")
    os.makedirs(video_dir, exist_ok=True)
    
    csv_dirname = os.path.dirname(args.csv_output)
    csv_basename = os.path.basename(args.csv_output)
    final_csv_path = os.path.join(csv_dirname, f"{prefix_str}{csv_basename}") if csv_dirname else f"{prefix_str}{csv_basename}"
    
    print(f"\n--- STARTING DETERMINISTIC INFERENCE BATCH SWEEP ---")
    print(f"Oracle Model: {args.model_path}")
    print(f"Configurations to evaluate: {args.n_test_array}")
    print(f"Games per configuration: {args.num_games_per_test}")
    
    for n_test in args.n_test_array:
        print(f"\nEvaluating N = {n_test} (x{args.num_games_per_test} environments)...")
        
        args.num_landmarks = n_test
        args.num_envs = n_test * args.num_games_per_test
        
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
                
                # Render grid layout for parallel games
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
                if len(step_data) >= 4 and isinstance(step_data[-1], dict) and "raw_obs" in step_data[-1]:
                    raw_obs = step_data[-1]["raw_obs"].clone().to(device)

        video_path = os.path.join(video_dir, f"{prefix_str}inference_N{n_test}.mp4")
        imageio.mimsave(video_path, video_frames, fps=15)
        print(f"Saved video to: {video_path}")

        metrics = tracker.get_summary()
        metrics = {"N_test": n_test, "Games_Sampled": args.num_games_per_test, **metrics}
        all_metrics.append(metrics)
        
        for k, v in metrics.items():
            print(f"  {k}: {v}")
            
        envs.close()

    if all_metrics:
        os.makedirs(os.path.dirname(os.path.abspath(final_csv_path)), exist_ok=True)
        keys = all_metrics[0].keys()
        
        with open(final_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_metrics)
            
        print(f"\n[Success] Batch inference metrics saved to: {final_csv_path}")

def harvest_imitation_data(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    video_dir = os.path.join(args.output_dir, "expert_videos")
    os.makedirs(video_dir, exist_ok=True)

    video_frames = []
    is_recording = False
    
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

                    episode_metrics = video_tracker.get_summary()
                    with open(metrics_path, "w") as f:
                        json.dump(episode_metrics, f, indent=2)
                        
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
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Import your architectures
from ppo_vmas_navigation_mappo import TransformerAgent, VMASVectorizedEnv

# ---------------------------------------------------------
# Helper: Load Model
# ---------------------------------------------------------
def load_oracle_model(args, envs, device):
    state_dim = envs.num_agents * np.array(envs.single_observation_space.shape).prod()
    oracle = TransformerAgent(
        envs.single_action_space, envs.single_observation_space.shape, 
        envs.num_agents, state_dim=state_dim, n_max=args.n_max * 2
    ).to(device)
    
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    keys_to_remove = [k for k in state_dict.keys() if "critic" in k or "embedding" in k]
    for k in keys_to_remove: del state_dict[k]
    
    oracle.load_state_dict(state_dict, strict=False)
    oracle.eval()
    return oracle

def get_action(oracle, obs):
    if hasattr(oracle, 'obs_normalizer'): obs_norm = oracle.obs_normalizer.normalize(obs)
    else: obs_norm = obs
        
    if hasattr(oracle, 'backbone'):
        valid_x, node_embeddings = oracle.backbone(obs_norm)[:2]
        actor_features = oracle.actor_mlp(node_embeddings[valid_x[:, 4] > 0.5])
    elif hasattr(oracle, 'transformer'):
        actor_features = oracle.actor(oracle._forward_actor_backbone(obs_norm))
    else:
        actor_features = oracle.actor(obs_norm)
        
    action = oracle.actor_mean(actor_features)
    return torch.clamp(action + torch.randn_like(action) * 0.025, -1.0, 1.0)

# ---------------------------------------------------------
# Tool 1: Spaghetti Plot (Qualitative Spatial Trajectories)
# ---------------------------------------------------------
def generate_spaghetti_plot(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    envs = VMASVectorizedEnv(args, args.seed, run_name="spaghetti", update_step=0)
    oracle = load_oracle_model(args, envs, device)
    
    pos_history = []
    goal_positions = None
    
    reset_data = envs.reset(seed=args.seed)
    obs = (reset_data[0] if isinstance(reset_data, tuple) else reset_data).to(device)
    raw_obs = (reset_data[1]["raw_obs"] if isinstance(reset_data, tuple) else obs).to(device)
    
    print(f"Tracing spatial trajectories for N={args.num_landmarks}...")
    with torch.no_grad():
        for step in tqdm(range(args.max_cycles)):
            # Extract current positions
            raw_flat = raw_obs.view(envs.num_games, envs.num_agents, -1)
            pos_history.append(raw_flat[0, :, 0:2].cpu().numpy()) # Save game 0
            
            if goal_positions is None:
                # Calculate absolute goal positions: Goal = Pos + To_Goal_Vector
                to_goal = raw_flat[0, :, 4:6].cpu().numpy()
                goal_positions = pos_history[0] + to_goal
                
            action = get_action(oracle, obs)
            step_data = envs.step(action)
            obs = step_data[0].to(device)
            raw_obs = step_data[-1]["raw_obs"].to(device) if "raw_obs" in step_data[-1] else obs

    envs.close()
    
    # Plotting
    pos_history = np.array(pos_history) # Shape: [Steps, N, 2]
    
    plt.figure(figsize=(10, 10))
    colors = plt.cm.get_cmap('hsv', args.num_landmarks)
    
    for i in range(args.num_landmarks):
        # Plot trajectory path
        plt.plot(pos_history[:, i, 0], pos_history[:, i, 1], color=colors(i), alpha=0.6, linewidth=1.5)
        # Plot start position
        plt.scatter(pos_history[0, i, 0], pos_history[0, i, 1], color=colors(i), marker='o', s=50, edgecolors='black')
        # Plot goal position
        plt.scatter(goal_positions[i, 0], goal_positions[i, 1], color=colors(i), marker='*', s=150, edgecolors='black')

    plt.title(f'Spatial Trajectories (N={args.num_landmarks})\nModel: {args.model_label}', fontsize=16)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'Spaghetti_Plot_{args.model_label}_N{args.num_landmarks}.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Spaghetti Plot to {out_path}")

# ---------------------------------------------------------
# Tool 2: Violin Plot (Arrival Time Distributions)
# ---------------------------------------------------------
def generate_violin_plot(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    all_arrival_times = []
    
    print(f"Collecting arrival distributions across {args.num_eval_episodes} episodes...")
    for ep in range(args.num_eval_episodes):
        ep_seed = args.seed + ep
        envs = VMASVectorizedEnv(args, ep_seed, run_name=f"violin_{ep}", update_step=0)
        oracle = load_oracle_model(args, envs, device)
        
        # Trackers
        convergence_steps = torch.full((envs.num_games, envs.num_agents), float('inf'), device=device)
        done_mask = torch.zeros((envs.num_games, envs.num_agents), dtype=torch.bool, device=device)
        
        reset_data = envs.reset(seed=ep_seed)
        obs = (reset_data[0] if isinstance(reset_data, tuple) else reset_data).to(device)
        raw_obs = (reset_data[1]["raw_obs"] if isinstance(reset_data, tuple) else obs).to(device)
        
        with torch.no_grad():
            for step in tqdm(range(args.max_cycles), leave=False):
                action = get_action(oracle, obs)
                
                raw_flat = raw_obs.view(envs.num_games, envs.num_agents, -1)
                to_goal = raw_flat[:, :, 4:6]
                goal_dists = torch.norm(to_goal, dim=-1)
                
                # Check 0.25 tolerance
                just_finished = (goal_dists < 0.25) & (~done_mask)
                convergence_steps[just_finished] = step + 1
                done_mask = done_mask | (goal_dists < 0.25)
                
                step_data = envs.step(action)
                obs = step_data[0].to(device)
                raw_obs = step_data[-1]["raw_obs"].to(device) if "raw_obs" in step_data[-1] else obs
        
        # Flatten and keep failures as max_cycles for plotting density
        times = convergence_steps.cpu().numpy().flatten()
        times[times == float('inf')] = args.max_cycles
        all_arrival_times.extend(times)
        envs.close()
        
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=all_arrival_times, inner="quartile", color="skyblue", cut=0)
    
    plt.title(f'Distribution of Steps to First Arrival (N={args.num_landmarks})\nModel: {args.model_label}', fontsize=14)
    plt.ylabel('Steps (Failures logged at Max Cycles)')
    plt.axhline(args.max_cycles, color='red', linestyle='--', alpha=0.5, label='Max Cycles (Failed to Arrive)')
    plt.legend()
    
    out_path = os.path.join(args.output_dir, f'Violin_Arrivals_{args.model_label}_N{args.num_landmarks}.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Violin Plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, choices=["spaghetti", "violin"], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-label", type=str, default="GAT", help="Name for the plot title (e.g., GAT, MLP)")
    parser.add_argument("--num-landmarks", type=int, default=105)
    parser.add_argument("--n-max", type=int, default=5)
    parser.add_argument("--max-cycles", type=int, default=350)
    parser.add_argument("--num-eval-episodes", type=int, default=10, help="Only used for violin plot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./plots")
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.env_id = "navigation"
    args.capture_video = False 
    args.reward_cheat = False
    args.num_envs = args.num_landmarks 
    
    if args.plot == "spaghetti": generate_spaghetti_plot(args)
    elif args.plot == "violin": generate_violin_plot(args)
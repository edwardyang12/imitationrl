import argparse
import os
import gc
import torch
import numpy as np
from tqdm import tqdm

# Import the architecture and environment wrapper directly from your training script
from ppo_vmas_navigation_gnn import GraphAgent, VMASVectorizedEnv

def parse_harvest_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved .pth oracle model file")
    parser.add_argument("--num-landmarks", type=int, default=7, help="Number of agents/landmarks to harvest (N)")
    parser.add_argument("--n-max", type=int, default=7, help="MUST MATCH TRAINING: The context window size")
    parser.add_argument("--num-trajectories", type=int, default=500000, help="Total step transitions to harvest")
    parser.add_argument("--chunk-size", type=int, default=100000, help="How many steps to hold in RAM before writing to disk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the harvesting environment")
    parser.add_argument("--output-dir", type=str, default="./expert_data", help="Directory to save the harvested numpy arrays")
    
    args = parser.parse_args()
    
    # Inject hidden args required by VMASVectorizedEnv initialization
    args.cuda = torch.cuda.is_available()
    args.env_id = "navigation"
    args.capture_video = False 
    args.reward_cheat = False
    
    # Vectorize environments based on the population size 
    args.num_envs = args.num_landmarks 
    
    return args

def harvest_vmas_dataset():
    args = parse_harvest_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- INITIALIZING EXPERT HARVESTING ---")
    print(f"Oracle Model: {args.model_path}")
    print(f"Population Size (N): {args.num_landmarks}")
    print(f"Target Steps: {args.num_trajectories} (Chunks of {args.chunk_size})")
    
    # 1. Initialize VMAS Environment
    envs = VMASVectorizedEnv(args, args.seed, run_name="harvest_run", update_step=0)
    
    # 2. Initialize and Load Oracle Agent
    n_max_nodes = args.n_max * 2
    oracle = GraphAgent(
        envs=envs, 
        n_max=n_max_nodes, 
        num_agents=args.num_landmarks
    ).to(device)
    
    print("Loading Oracle weights...")
    oracle.load_state_dict(torch.load(args.model_path, map_location=device))
    oracle.eval() # Freeze layers for evaluation rolling
    
    # Data buffers
    expert_obs = []
    expert_actions = []
    chunk_idx = 0
    
    # Initial Environment Reset
    reset_data = envs.reset(seed=args.seed)
    obs = reset_data[0] if isinstance(reset_data, tuple) else reset_data
    obs = obs.clone().to(device)
    
    print("Starting data collection...")
    with torch.no_grad():
        for step in tqdm(range(args.num_trajectories)):
            # 1. Forward pass through the trained oracle policy
            action, _, _, _ = oracle.get_action_and_value(obs)
            
            # Clip actions to valid physical simulation limits
            clipped_action = torch.clamp(action, -1.0, 1.0)
            
            # 2. Append un-pushed tensor states to RAM logs (convert to CPU NumPy arrays)
            expert_obs.append(obs.cpu().numpy())
            expert_actions.append(clipped_action.cpu().numpy())
            
            # 3. Environment Step execution
            step_data = envs.step(clipped_action)
            obs = step_data[0].clone().to(device)
            
            # 4. Memory check & safe array dumping
            if len(expert_obs) >= args.chunk_size:
                obs_array = np.vstack(expert_obs)
                act_array = np.vstack(expert_actions)
                
                obs_save_path = os.path.join(args.output_dir, f"obs_N{args.num_landmarks}_part{chunk_idx}.npy")
                act_save_path = os.path.join(args.output_dir, f"actions_N{args.num_landmarks}_part{chunk_idx}.npy")
                
                np.save(obs_save_path, obs_array)
                np.save(act_save_path, act_array)
                print(f"\n[Memory Check] Saved chunk {chunk_idx}. Flushing RAM...")
                
                # Free memory overhead allocations explicitly
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
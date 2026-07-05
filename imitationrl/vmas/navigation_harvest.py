import argparse
import os
import gc
import torch
import numpy as np
import imageio
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
    
    # 2. Initialize and Load Oracle Agent
    n_max_nodes = args.n_max * 2
    oracle = GraphAgent(
        envs=envs, 
        n_max=n_max_nodes, 
        num_agents=args.num_landmarks
    ).to(device)
    
    print("Loading Oracle weights...")
    oracle.load_state_dict(torch.load(args.model_path, map_location=device))
    oracle.eval() # Freeze layers and set to evaluation mode
    
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

            if step % args.video_interval == 0:
                is_recording = True
                video_frames = []
                print(f"\n[Video] Starting recording at step {step}...")

            # --- DETERMINISTIC FORWARD PASS ---
            # 1. Normalize the incoming observations if an obs_normalizer exists
            if hasattr(oracle, 'obs_normalizer'):
                obs_norm = oracle.obs_normalizer.normalize(obs)
            else:
                obs_norm = obs
            
            # 2. Pass through the GNN backbone to get valid nodes and embeddings
            # (Matches the backbone signature: returns valid_x, node_embeddings, and optionally batch/etc.)
            backbone_outputs = oracle.backbone(obs_norm)
            valid_x = backbone_outputs[0]
            node_embeddings = backbone_outputs[1]
            
            # 3. Filter for active ego-agents taking actions
            agent_mask = valid_x[:, 4] > 0.5
            
            # 4. Extract MLP features and map directly to deterministic action means (no sampling!)
            actor_features = oracle.actor_mlp(node_embeddings[agent_mask])
            deterministic_action = oracle.actor_mean(actor_features)
            
            # Clip actions to valid physical simulation limits [-1, 1]
            noise = torch.randn_like(deterministic_action) * 0.025
            clipped_action = torch.clamp(deterministic_action + noise, -1.0, 1.0)

            # --- RENDER FRAME IF RECORDING ---
            if is_recording:
                # Render the frame directly from the underlying VMAS engine
                frame = envs.env.render(mode="rgb_array", env_index=0, agent_index_focus=None)
                if isinstance(frame, list):
                    frame = frame[0]
                video_frames.append(frame)
                
                # If we've collected enough frames, save the video file
                if len(video_frames) >= args.max_cycles:
                    video_path = os.path.join(video_dir, f"expert_step_{step}.mp4")
                    imageio.mimsave(video_path, video_frames, fps=15)
                    print(f"[Video] Saved sample behavior to {video_path}")
                    video_frames = []
                    is_recording = False
            
            # 5. Append un-pushed raw observations and continuous actions to RAM logs
            expert_obs.append(obs.cpu().numpy())
            expert_actions.append(clipped_action.cpu().numpy())
            
            # 6. Environment Step execution
            step_data = envs.step(clipped_action)
            obs = step_data[0].clone().to(device)
            
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
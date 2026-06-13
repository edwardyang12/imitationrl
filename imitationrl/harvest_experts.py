import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
import gc
import gymnasium as gym
from torch.distributions.categorical import Categorical
import scipy

# Import your environment builder and Agent class from your main script
from mpe.ppo_pettingzoo_ma_atari_mappo import build_environments, Agent, parse_args

def harvest_dataset(N, num_trajectories=500000, chunk_size=100000):
    args = parse_args()
    args.num_landmarks = N
    args.reward_cheat = False # Turn off the cheat; we just want the physical state
    args.env_id = "simple_spread_v3"
    
    # Disable the default video wrapper in build_environments so we can customize it here
    args.capture_video = False 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Build strict, un-hacked environment
    run_name = f"harvest_N{N}"
    envs, num_agents_per_game, num_games = build_environments(args, run_name, args.seed, current_local_ratio=0.5)
    
    # --- CUSTOM VIDEO RECORDER ---
    # Wrap the environment to record a video every 100 episodes
    envs = gym.wrappers.vector.RecordVideo(
        envs, 
        f"expert_videos/{run_name}", 
        episode_trigger=lambda x: x % 100 == 0, # Change this number to record more/less often
        name_prefix=f"expert_behavior_N{N}"
    )
    
    # 2. Load Oracle
    state_dim = (num_agents_per_game * np.array(envs.single_observation_space.shape).prod()) + args.num_landmarks
    oracle = Agent(envs, num_agents_per_game, state_dim).to(device)
    oracle.load_state_dict(torch.load(f"models/simple_spread_v3__ppo_pettingzoo_ma_atari_mappo__1__1780740016/1265_model.pth",
                                      map_location=device), strict=True) # Ensure path points to your saved model
    oracle.eval() # Lock batchnorm/dropout

    expert_obs = []
    expert_actions = []
    chunk_idx = 0

    next_obs = torch.Tensor(envs.reset(seed=args.seed)[0]).to(device)
    
    # --- FIX INITIALIZATION: Set up tracking for the Oracle ---
    current_assignments = torch.arange(args.num_landmarks).unsqueeze(0).repeat(num_games, 1).to(device)
    needs_assignment = torch.ones(num_games, dtype=torch.bool).to(device)

    # Perform initial assignment for the first step
    dist_cpu = torch.norm(
        next_obs.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks].view(num_games, num_agents_per_game, args.num_landmarks, 2), 
        dim=-1
    ).cpu().numpy()
    
    for g in range(num_games):
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_cpu[g])
        current_assignments[g] = torch.tensor(col_ind, device=device)
    needs_assignment.fill_(False)
    # ---------------------------------------------------------
    
    print(f"Harvesting {num_trajectories} steps for N={N} in chunks of {chunk_size}...")
    os.makedirs("expert_data", exist_ok=True)

    with torch.no_grad():
        for step in tqdm(range(num_trajectories)):
            
            # --- THE FIX: PERMUTE OBSERVATIONS FOR THE ORACLE ---
            oracle_obs = next_obs.clone()
            obs_resh_temp = oracle_obs.view(num_games, num_agents_per_game, -1)
            landmarks_segment = obs_resh_temp[:, :, 4:4+2*args.num_landmarks].view(num_games, num_agents_per_game, args.num_landmarks, 2)

            base_indices = torch.arange(args.num_landmarks, device=device).expand(num_games, num_agents_per_game, -1).clone()
            base_indices[:, :, 0] = current_assignments
            
            g_idx = torch.arange(num_games, device=device).view(-1, 1)
            a_idx = torch.arange(num_agents_per_game, device=device).view(1, -1)
            base_indices[g_idx, a_idx, current_assignments] = 0

            gather_indices = base_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
            swapped_landmarks = torch.gather(landmarks_segment, dim=2, index=gather_indices)

            obs_resh_temp[:, :, 4:4+2*args.num_landmarks] = swapped_landmarks.view(num_games, num_agents_per_game, -1)
            oracle_obs = obs_resh_temp.view(-1, oracle.obs_dim)
            # ----------------------------------------------------
            
            # 1. Use the permuted observation for the Oracle forward pass
            normalized_obs = oracle.obs_normalizer.normalize(oracle_obs)
            obs_reshaped = normalized_obs.view(-1, num_agents_per_game, oracle.obs_dim)
            
            actions_list = []
            for i in range(num_agents_per_game):
                logits = oracle.actors[i](obs_reshaped[:, i, :])
                
                temperature = 1.25 
                probs = Categorical(logits=logits / temperature)
                sampled_action = probs.sample() 
                actions_list.append(sampled_action)
            
            action = torch.stack(actions_list, dim=1).view(-1)
            
            # 2. Save the UNPERMUTED state to the dataset
            expert_obs.append(next_obs.cpu().numpy())
            expert_actions.append(action.cpu().numpy())

            if len(expert_obs) >= chunk_size:
                dataset = {
                    "observations": np.vstack(expert_obs), 
                    "actions": np.concatenate(expert_actions) 
                }
                save_path = f"expert_data/expert_N{N}_part{chunk_idx}.pkl"
                with open(save_path, "wb") as f:
                    pickle.dump(dataset, f)
                print(f"\n[Memory Check] Saved {save_path}. Clearing RAM...")
                
                expert_obs.clear()
                expert_actions.clear()
                gc.collect()
                chunk_idx += 1
            
            # 3. Step the environment and trigger reassignments on reset
            step_data = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(step_data[0] if len(step_data) == 5 else step_data[0]).to(device)

            resets = np.logical_or(step_data[2], step_data[3]) if len(step_data) == 5 else step_data[2]
            game_resets = torch.Tensor(resets).to(device).view(num_games, num_agents_per_game)[:, 0].bool()
            needs_assignment = needs_assignment | game_resets
            
            if needs_assignment.any():
                dist_cpu = torch.norm(
                    next_obs.view(num_games, num_agents_per_game, -1)[:, :, 4:4+2*args.num_landmarks].view(num_games, num_agents_per_game, args.num_landmarks, 2), 
                    dim=-1
                ).cpu().numpy()
                
                for g in range(num_games):
                    if needs_assignment[g]:
                        row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_cpu[g])
                        current_assignments[g] = torch.tensor(col_ind, device=device)
                        
                needs_assignment.fill_(False)

    # 3. Save to disk
    if len(expert_obs) > 0:
        dataset = {
            "observations": np.vstack(expert_obs), 
            "actions": np.concatenate(expert_actions) 
        }
        with open(f"expert_data/expert_N{N}_part{chunk_idx}.pkl", "wb") as f:
            pickle.dump(dataset, f)
        print(f"\n[Memory Check] Saved final remainder chunk to part{chunk_idx}.")

    print(f"Successfully finished harvesting N={N}!")
    envs.close()

if __name__ == "__main__":
    # Ensure you update the model path inside the function before running
    for N in [9]:
        harvest_dataset(N, num_trajectories=500000, chunk_size=200000)
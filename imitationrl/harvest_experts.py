import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
import gc
import gymnasium as gym
from torch.distributions.categorical import Categorical

# Import your environment builder and Agent class from your main script
from ppo_pettingzoo_ma_atari_mappo import build_environments, Agent, parse_args

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
    oracle.load_state_dict(torch.load(f"models/simple_spread_v3__ppo_pettingzoo_ma_atari_mappo__1__1777972083/1139_model.pth")) # Ensure path points to your saved model
    oracle.eval() # Lock batchnorm/dropout

    expert_obs = []
    expert_actions = []
    chunk_idx = 0

    next_obs = torch.Tensor(envs.reset(seed=args.seed)[0]).to(device)
    
    print(f"Harvesting {num_trajectories} steps for N={N} in chunks of {chunk_size}...")
    os.makedirs("expert_data", exist_ok=True)

    with torch.no_grad():
        for step in tqdm(range(num_trajectories)):
            
            # --- FIX 1: THE PRECISION ALIGNMENT ---
            # We must pass the exact normalized matrix the Oracle was trained on
            normalized_obs = oracle.obs_normalizer.normalize(next_obs)
            obs_reshaped = normalized_obs.view(-1, num_agents_per_game, oracle.obs_dim)
            
            actions_list = []
            for i in range(num_agents_per_game):
                logits = oracle.actors[i](obs_reshaped[:, i, :])
                
                # PURE EXPLOITATION: No categorical sampling
                # deterministic_action = torch.argmax(logits, dim=-1) 
                # actions_list.append(deterministic_action)

                # --- FIX 2: TEMPERATURE-SCALED JITTER ---
                # A fully converged N=5 policy has such extreme logits that sample() 
                # practically acts like argmax. We divide by a temperature to slightly 
                # soften the distribution, guaranteeing the micro-jitter needed to break deadlocks.
                temperature = 1.25 
                probs = Categorical(logits=logits / temperature)
                sampled_action = probs.sample() 
                
                actions_list.append(sampled_action)
            
            action = torch.stack(actions_list, dim=1).view(-1)
            
            # Save the RAW (unnormalized) observations so the student network 
            # learns to master the native environment.
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
                
                # Nuke the lists from RAM and force garbage collection
                expert_obs.clear()
                expert_actions.clear()
                gc.collect()
                
                chunk_idx += 1
            
            step_data = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(step_data[0] if len(step_data) == 5 else step_data[0]).to(device)

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
    for N in [6]:
        harvest_dataset(N, num_trajectories=500000, chunk_size=200000)
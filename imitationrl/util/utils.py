import numpy as np
import pickle
import glob
import os
from tqdm import tqdm

def convert_pkl_to_npy(n_values):
   
    for N in n_values:
        print(f"Converting N={N} data...")
        file_pattern = f"D:/Edward/imitationrl/expert_data/expert_N{N}*.pkl"
        part_files = glob.glob(file_pattern)
        
        if not part_files:
            print(f"  Warning: No files found for N={N}")
            continue
            
        all_obs = []
        all_actions = []
        
        for file_path in tqdm(part_files, desc=f"Loading N={N} Pickles"):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                all_obs.append(data["observations"])
                all_actions.append(data["actions"].flatten())
                
        # Stack into single numpy arrays
        stacked_obs = np.vstack(all_obs).astype(np.float32)
        stacked_actions = np.concatenate(all_actions).astype(np.int64)
        
        # Save as .npy formats
        np.save(f"D:/Edward/imitationrl/expert_data/obs_N{N}.npy", stacked_obs)
        np.save(f"D:/Edward/imitationrl/expert_data/actions_N{N}.npy", stacked_actions)
        
        print(f"  Saved N={N}: Obs shape {stacked_obs.shape}, Actions shape {stacked_actions.shape}")

if __name__ == "__main__":
    convert_pkl_to_npy(n_values=[3])
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from tqdm import tqdm
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class VectorizedBatchDataset(Dataset):
    def __init__(self, n_values, n_max=10, batch_size=32768):
        self.n_max = n_max
        self.batch_size = batch_size
        self.target_obs_dim = (4 * 6 * n_max) + n_max
        
        self.batches = []
        self.worker_mmaps = {}
        
        print("Calculating Vectorized Dataset Dimensions...")
        for N in n_values:
            obs_path = f"D:/Edward/imitationrl/expert_data/obs_N{N}.npy"
            act_path = f"D:/Edward/imitationrl/expert_data/actions_N{N}.npy"
            
            if not os.path.exists(obs_path) or not os.path.exists(act_path):
                continue
                
            # Calculate how many full batches fit in this N file
            temp_act = np.load(act_path, mmap_mode='r')
            num_batches = len(temp_act) // self.batch_size
            
            for i in range(num_batches):
                self.batches.append({
                    'N': N,
                    'start_idx': i * self.batch_size,
                    'obs_path': obs_path,
                    'act_path': act_path
                })
            print(f"  Mapped N={N}: {num_batches} full batches.")
            
    def __len__(self):
        # Length is now the number of BATCHES, not individual samples
        return len(self.batches)
        
    def __getitem__(self, idx):
        batch_meta = self.batches[idx]
        N = batch_meta['N']
        start_idx = batch_meta['start_idx']
        end_idx = start_idx + self.batch_size
        
        if N not in self.worker_mmaps:
            self.worker_mmaps[N] = {
                'obs': np.load(batch_meta['obs_path'], mmap_mode='r'),
                'act': np.load(batch_meta['act_path'], mmap_mode='r')
            }
            
        mmap_data = self.worker_mmaps[N]
        raw_obs_batch = np.array(mmap_data['obs'][start_idx:end_idx]) 
        action_batch = np.array(mmap_data['act'][start_idx:end_idx])
        
        B = raw_obs_batch.shape[0]
        
        # 1. VECTORIZED UNPACKING
        frames_part = raw_obs_batch[:, :-N] 
        indicator_part = raw_obs_batch[:, -N:]
        frames = frames_part.reshape(B, 4, 6 * N)
        
        # 2. ISOLATE ENTITIES
        self_data = frames[:, :, 0:4] # (B, 4, 4)
        landmarks = frames[:, :, 4 : 4+2*N].reshape(B, 4, N, 2)
        
        other_pos_start = 4 + 2*N
        other_pos = frames[:, :, other_pos_start : other_pos_start + 2*(N-1)].reshape(B, 4, N-1, 2)
        
        other_vel_start = other_pos_start + 2*(N-1)
        other_vel = frames[:, :, other_vel_start : 6*N].reshape(B, 4, N-1, 2)
        
        # 3. ROW-WISE UNIQUE PERMUTATIONS
        # np.argsort on random matrices generates unique random choices per row instantly
        landmark_slots = np.argsort(np.random.rand(B, self.n_max), axis=1)[:, :N]
        agent_slots = np.argsort(np.random.rand(B, self.n_max - 1), axis=1)[:, :N - 1]
        
        # 4. PRE-ALLOCATE ZERO-PADDED DESTINATIONS
        dest_landmarks = np.zeros((B, 4, self.n_max, 2), dtype=np.float32)
        dest_other_pos = np.zeros((B, 4, self.n_max - 1, 2), dtype=np.float32)
        dest_other_vel = np.zeros((B, 4, self.n_max - 1, 2), dtype=np.float32)
        
        # 5. ADVANCED INDEXING ASSIGNMENT
        b_idx = np.arange(B)[:, None, None]
        f_idx = np.arange(4)[None, :, None]
        
        dest_landmarks[b_idx, f_idx, landmark_slots[:, None, :]] = landmarks
        dest_other_pos[b_idx, f_idx, agent_slots[:, None, :]] = other_pos
        dest_other_vel[b_idx, f_idx, agent_slots[:, None, :]] = other_vel
        
        # 6. FLATTEN AND RECOMBINE FRAMES
        dest_landmarks_flat = dest_landmarks.reshape(B, 4, self.n_max * 2)
        dest_other_pos_flat = dest_other_pos.reshape(B, 4, (self.n_max - 1) * 2)
        dest_other_vel_flat = dest_other_vel.reshape(B, 4, (self.n_max - 1) * 2)
        
        padded_frames = np.concatenate([
            self_data, dest_landmarks_flat, dest_other_pos_flat, dest_other_vel_flat
        ], axis=2) 
        
        # 7. ROW-WISE INDICATOR PERMUTATION
        original_agent_idx = np.argmax(indicator_part, axis=1)
        available_id_slots = np.argsort(np.random.rand(B, self.n_max), axis=1)[:, :N]
        
        new_agent_idx = available_id_slots[np.arange(B), original_agent_idx]
        
        padded_indicator = np.zeros((B, self.n_max), dtype=np.float32)
        padded_indicator[np.arange(B), new_agent_idx] = 1.0
        
        # 8. FINAL STITCH AND INTRA-BATCH SHUFFLE
        padded_frames_flat = padded_frames.reshape(B, 4 * 6 * self.n_max)
        final_obs = np.concatenate([padded_frames_flat, padded_indicator], axis=1)
        
        shuffle_idx = np.random.permutation(B)
        final_obs = final_obs[shuffle_idx]
        action_batch = action_batch[shuffle_idx]
        
        return torch.Tensor(final_obs), torch.tensor(action_batch, dtype=torch.long)

class StudentActor(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.LayerNorm(1024), nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024), nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
    def forward(self, x):
        return self.network(x)

def train_behavioral_cloning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_max = 6
    batch_size = 32768
    n_values_to_train = [3, 4]
    
    # Initialize the Vectorized Dataset
    full_dataset = VectorizedBatchDataset(n_values=n_values_to_train, n_max=n_max, batch_size=batch_size)
    
    # Split the BATCHES, not the individual items
    total_batches = len(full_dataset)
    val_batches = int(0.2 * total_batches)
    train_batches = total_batches - val_batches
    
    generator = torch.Generator().manual_seed(42) 
    train_dataset, val_dataset = random_split(full_dataset, [train_batches, val_batches], generator=generator)
    
    print(f"Dataset Split: {train_batches} Train Batches | {val_batches} Validation Batches")
    
    # Setting batch_size=None tells the DataLoader that the Dataset is already outputting full batches
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
    
    obs_dim = full_dataset.target_obs_dim 
    num_actions = 5 
    
    student = StudentActor(obs_dim, num_actions).to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    best_val_accuracy = 0.0 
    
    print(f"Starting Behavioral Cloning (N_max={n_max}) for {epochs} epochs...")
    
    for epoch in range(epochs):
        student.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch_obs, batch_actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_obs, batch_actions = batch_obs.to(device), batch_actions.to(device)
            
            optimizer.zero_grad()
            logits = student(batch_obs)
            
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            train_correct += (predictions == batch_actions).sum().item()
            train_total += batch_actions.size(0)
            
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        student.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad(): 
            for batch_obs, batch_actions in val_loader:
                batch_obs, batch_actions = batch_obs.to(device), batch_actions.to(device)
                
                logits = student(batch_obs)
                loss = criterion(logits, batch_actions)
                
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                val_correct += (predictions == batch_actions).sum().item()
                val_total += batch_actions.size(0)
                
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% || Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(student.state_dict(), f"student_bc_best_nmax{n_max}.pt")
            print(f"  -> New best model saved! (Val Acc: {best_val_accuracy:.2f}%)")

if __name__ == "__main__":
    train_behavioral_cloning()
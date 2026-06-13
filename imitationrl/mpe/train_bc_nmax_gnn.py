import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import GATConv, global_mean_pool
import numpy as np
import os
from tqdm import tqdm
import random
from torch_geometric.nn import radius_graph

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MultiHeadGATBackbone(nn.Module):
    def __init__(self, n_max, feature_dim=8, hidden_dim=64, out_dim=128, heads=4):
        super().__init__()
        self.n_max = n_max
        self.feature_dim = feature_dim
        
        self.gat1 = GATConv(feature_dim, hidden_dim, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=1)
        self.gat3 = GATConv(hidden_dim * heads, out_dim, heads=heads, concat=False, edge_dim=1)
        
        self.skip_proj = nn.Linear(feature_dim, hidden_dim * heads)
        self.elu = nn.ELU()

    def _build_dynamic_graph(self, x_flat):
        B = x_flat.shape[0]
        device = x_flat.device
        
        # Reshape using n_max (which is passed as total_nodes)
        x_padded = x_flat.view(B, self.n_max, self.feature_dim)
        active_mask = x_padded[:, :, 6] > 0.5
        
        # 1. Strip out the ghost nodes BEFORE building the graph
        valid_x = x_padded[active_mask] 
        
        # 2. Build the batch index for PyTorch Geometric
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(B, self.n_max)
        valid_batch = batch_indices[active_mask] 
        
        # 3. Dynamic radius graph (matches PPO exactly)
        edge_index = radius_graph(valid_x[:, :2], r=1.0, batch=valid_batch, loop=False)
        
        # 4. Physics Extraction
        row, col = edge_index
        diff = valid_x[row, :2] - valid_x[col, :2]
        distances = torch.sqrt((diff ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        
        return valid_x, edge_index, distances, valid_batch

    def forward(self, x_flat):
        valid_x, edge_index, edge_attr, valid_batch = self._build_dynamic_graph(x_flat)
        
        res = self.skip_proj(valid_x)
        
        h = self.gat1(valid_x, edge_index, edge_attr=edge_attr)
        h = self.elu(h) + res 
        
        h2 = self.gat2(h, edge_index, edge_attr=edge_attr)
        h2 = self.elu(h2) + h 
        
        valid_embeddings = self.gat3(h2, edge_index, edge_attr=edge_attr) 
        
        # Return only valid nodes. The StudentActor handles the ego extraction.
        return valid_x, valid_embeddings

class VectorizedBatchDataset(Dataset):
    def __init__(self, n_values, max_n=4, batch_size=32768):
        self.batch_size = batch_size
        
        # Define the structural capacity of the Graph
        self.max_landmarks = max_n
        self.max_other_agents = max_n - 1
        self.total_nodes = 1 + self.max_landmarks + self.max_other_agents # Equals 2 * max_n
        
        # Flattened dimension for the DataLoader (e.g., 8 nodes * 8 features = 64)
        self.target_obs_dim = self.total_nodes * 8 
        
        self.batches = []
        self.worker_mmaps = {}
        
        print(f"Graph Capacity: {self.total_nodes} Total Nodes (1 Ego, {self.max_landmarks} LMs, {self.max_other_agents} Others)")
        for N in n_values:
            obs_path = f"D:/Edward/imitationrl/expert_data/obs_N{N}.npy"
            act_path = f"D:/Edward/imitationrl/expert_data/actions_N{N}.npy"
            
            if not os.path.exists(obs_path) or not os.path.exists(act_path):
                continue
                
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
        frames = raw_obs_batch[:, :-N] 
        self_data = frames[:, 0:4] # (B, 4)
        landmarks = frames[:, 4 : 4+2*N].reshape(B, N, 2)
        other_pos_start = 4 + 2*N
        other_pos = frames[:, other_pos_start : other_pos_start + 2*(N-1)].reshape(B, N-1, 2)
        
        # 2. DYNAMIC BATCH TAGS (1 for Ego, N-1 for Other Agents)
        tags = np.random.rand(B, N)
        
        # 3. PRE-ALLOCATE ZERO-PADDED GRAPH
        dest_nodes = np.zeros((B, self.total_nodes, 8), dtype=np.float32)
        b_idx = np.arange(B)[:, None]
        
        # --- Node 0: Ego Agent ---
        dest_nodes[:, 0, 0:2] = 0.0               # Ego relative position is always [0,0]
        dest_nodes[:, 0, 2:4] = self_data[:, 0:2] # Extract [vel_x, vel_y]
        dest_nodes[:, 0, 4] = 1.0  # is_self
        dest_nodes[:, 0, 6] = 1.0  # is_active
        dest_nodes[:, 0, 7] = tags[:, 0]
        
        # --- Nodes 1 to Max Landmarks: Shuffled Active Landmarks ---
        # Pick N random slots out of max_landmarks
        lm_slots = np.argsort(np.random.rand(B, self.max_landmarks), axis=1)[:, :N]
        lm_slots += 1 # Shift index to start after Ego
        
        dest_nodes[b_idx, lm_slots, 0:2] = landmarks
        dest_nodes[b_idx, lm_slots, 5] = 1.0  # is_landmark
        dest_nodes[b_idx, lm_slots, 6] = 1.0  # is_active
        
        # --- Remaining Nodes: Shuffled Active Other Agents ---
        # Pick N-1 random slots out of max_other_agents
        oa_slots = np.argsort(np.random.rand(B, self.max_other_agents), axis=1)[:, :N-1]
        oa_slots += (1 + self.max_landmarks) # Shift index to start after Landmarks
        
        dest_nodes[b_idx, oa_slots, 0:2] = other_pos
        dest_nodes[b_idx, oa_slots, 6] = 1.0  # is_active
        dest_nodes[b_idx, oa_slots, 7] = tags[:, 1:]
        
        # 4. FLATTEN AND SHUFFLE BATCH
        final_obs = dest_nodes.reshape(B, self.total_nodes * 8)
        
        shuffle_idx = np.random.permutation(B)
        final_obs = final_obs[shuffle_idx]
        action_batch = action_batch[shuffle_idx]
        
        return torch.from_numpy(final_obs).float(), torch.tensor(action_batch, dtype=torch.long)

class StudentGraphActor(nn.Module):
    def __init__(self, n_max, num_actions):
        super().__init__()
        self.backbone = MultiHeadGATBackbone(n_max=n_max, feature_dim=8)
        gat_out_dim = 128
        
        self.actor_mlp = nn.Sequential(
            nn.Linear(gat_out_dim, 128),
            nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
    def forward(self, x_flat):
        valid_x, valid_embeddings = self.backbone(x_flat)
        
        # Isolate only the specific agent taking actions
        agent_mask = valid_x[:, 4] > 0.5
        agent_embeddings = valid_embeddings[agent_mask]
        
        logits = self.actor_mlp(agent_embeddings)
        return logits

def train_behavioral_cloning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = torch.cuda.amp.GradScaler()
    
    n_max = 7
    batch_size = 8192
    n_values_to_train = [7]
    
    # Initialize the Vectorized Dataset
    full_dataset = VectorizedBatchDataset(n_values=n_values_to_train, max_n=n_max, batch_size=batch_size)
    
    # Split the BATCHES, not the individual items
    total_batches = len(full_dataset)
    val_batches = int(0.2 * total_batches)
    train_batches = total_batches - val_batches
    
    generator = torch.Generator().manual_seed(42) 
    train_dataset, val_dataset = random_split(full_dataset, [train_batches, val_batches], generator=generator)
    
    print(f"Dataset Split: {train_batches} Train Batches | {val_batches} Validation Batches")
    
    # Setting batch_size=None tells the DataLoader that the Dataset is already outputting full batches
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=seed_worker)
    
    num_actions = 5
    
    total_nodes = full_dataset.total_nodes
    
    student = StudentGraphActor(total_nodes, num_actions).to(device)
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

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = student(batch_obs)
                loss = criterion(logits, batch_actions)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
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
            torch.save(student.state_dict(), f"gnn_student_bc_best_nmax{n_max}.pt")
            print(f"  -> New best model saved! (Val Acc: {best_val_accuracy:.2f}%)")

if __name__ == "__main__":
    train_behavioral_cloning()
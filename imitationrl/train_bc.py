import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import glob
from tqdm import tqdm

class ExpertDataset(Dataset):
    def __init__(self, n_values=[3]): # You can expand this to [3, 4, 5] when ready
        all_obs = []
        all_actions = []
        
        for N in n_values:
            # Use glob to find all parts for this N value
            file_pattern = f"expert_data/expert_N{N}*.pkl"
            part_files = glob.glob(file_pattern)
            
            if not part_files:
                print(f"Warning: No files found for N={N}")
                continue
                
            for file_path in part_files:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    all_obs.append(data["observations"])
                    all_actions.append(data["actions"].flatten())
                
        self.obs = torch.Tensor(np.vstack(all_obs))
        self.actions = torch.LongTensor(np.concatenate(all_actions))
        
    def __len__(self):
        return len(self.actions)
        
    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]

class StudentActor(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
    def forward(self, x):
        return self.network(x)

def train_behavioral_cloning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Expert Datasets...")
    full_dataset = ExpertDataset(n_values=[3])
    
    # --- 1. TRAIN / VALIDATION SPLIT ---
    # Standard 80/20 split
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # Use a fixed generator seed for reproducible splits if desired
    generator = torch.Generator().manual_seed(42) 
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=32768, shuffle=True, num_workers=4, pin_memory=True)
    # Validation loader doesn't need to be shuffled
    val_loader = DataLoader(val_dataset, batch_size=32768, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset Split: {train_size} Train | {val_size} Validation")
    
    # Get dimensions dynamically from the full dataset
    obs_dim = full_dataset.obs.shape[1]
    num_actions = len(torch.unique(full_dataset.actions))
    
    student = StudentActor(obs_dim, num_actions).to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 50
    best_val_accuracy = 0.0 # Track the best model
    
    print(f"Starting Behavioral Cloning for {epochs} epochs...")
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        student.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
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
        
        # --- VALIDATION PHASE ---
        student.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # Disable gradient tracking for speed and memory
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
        
        # --- BEST MODEL CHECKPOINTING ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(student.state_dict(), "student_bc_best.pt")
            print(f"  -> New best model saved! (Val Acc: {best_val_accuracy:.2f}%)")

    # Save the final model just in case, but usually student_bc_best.pt is the one you want
    torch.save(student.state_dict(), "student_bc_final.pt")
    print("Pre-training complete.")

if __name__ == "__main__":
    train_behavioral_cloning()
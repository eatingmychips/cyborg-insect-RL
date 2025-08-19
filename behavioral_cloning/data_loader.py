import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import pickle

class DemonstrationDataset(Dataset):
    """Dataset class for behavioral cloning demonstrations."""
    
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)  # For discrete actions
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class DemonstrationDataLoader:
    """Loads and preprocesses demonstration data from CSV files."""
    
    def __init__(self, state_columns: list, action_column: str, 
                 normalize: bool = True, train_split: float = 0.8):
        self.state_columns = state_columns
        self.action_column = action_column
        self.normalize = normalize
        self.train_split = train_split
        self.scaler = StandardScaler() if normalize else None
        
    def load_csv_files(self, csv_paths: list) -> Tuple[np.ndarray, np.ndarray]:
        """Load and concatenate multiple CSV files."""
        all_data = []
        for path in csv_paths:
            df = pd.read_csv(path)
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        return self._process_dataframe(combined_df)
    
    def _process_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract states and actions from dataframe."""
        # Extract state features
        states = df[self.state_columns].values
        
        # Extract actions (assuming integer encoding for discrete actions)
        actions = df[self.action_column].values
        
        # Normalize states if requested
        if self.normalize:
            states = self.scaler.fit_transform(states)
        
        return states, actions
    
    def create_dataloaders(self, csv_paths: list, batch_size: int = 32, 
                          shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Create train/validation dataloaders."""
        states, actions = self.load_csv_files(csv_paths)
        
        # Train/validation split
        n_train = int(len(states) * self.train_split)
        
        # Shuffle data before splitting
        if shuffle:
            indices = np.random.permutation(len(states))
            states = states[indices]
            actions = actions[indices]
        
        train_states, val_states = states[:n_train], states[n_train:]
        train_actions, val_actions = actions[:n_train], actions[n_train:]
        
        # Create datasets
        train_dataset = DemonstrationDataset(train_states, train_actions)
        val_dataset = DemonstrationDataset(val_states, val_actions)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def save_scaler(self, path: str):
        """Save the fitted scaler for consistent preprocessing in RL."""
        if self.scaler:
            with open(path, 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def load_scaler(self, path: str):
        """Load a previously fitted scaler."""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)

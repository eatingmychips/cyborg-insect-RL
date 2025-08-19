import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm

class BCTrainer:
    """Behavioral Cloning trainer for policy networks."""
    
    def __init__(self, policy_network: nn.Module, learning_rate: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.policy = policy_network.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()  # For discrete actions
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.policy.train()
        total_loss = 0.0
        
        for states, actions in tqdm(train_loader, desc="Training"):
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.policy(states)
            loss = self.criterion(predictions, actions)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.policy.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                predictions = self.policy(states)
                loss = self.criterion(predictions, actions)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(predictions.data, 1)
                total += actions.size(0)
                correct += (predicted == actions).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, patience: int = 10, 
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']
            val_accuracy = val_metrics['accuracy']
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Logging
            logging.info(f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val Acc: {val_accuracy:.2f}%")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss
        }
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])

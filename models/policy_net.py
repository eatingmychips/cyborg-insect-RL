import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    """Simple feedforward policy network for discrete actions."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [128, 64]):
        super(PolicyNetwork, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Helps prevent overfitting
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - CrossEntropyLoss handles this)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass - returns action logits."""
        return self.network(x)
    
    def get_action(self, state, deterministic=False):
        """Get action from state (useful for evaluation/RL later)."""
        self.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.multinomial(probs, 1)
            
            return action.item()

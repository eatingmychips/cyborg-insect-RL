import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

# --- Load your environment and BC network ---
from env.cyborg_env import CyborgInsectEnv
from models.policy_net import PolicyNetwork

# Set up the environment (same as before)
image_height = 800
image_length = 1200

t = np.linspace(0, 2 * np.pi, 1000)
x = 100 + (image_length - 150) * t / (2 * np.pi)
y = 400 + (image_height/2 - 100) * np.sin(t)
path = np.column_stack((x, y))
path_int = np.round(path).astype(np.int32)

env = CyborgInsectEnv(path=path_int)


# Load the scaler for consistent preprocessing
with open('data/checkpoints/state_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Environment state processing function
def process_state_for_bc(env_state, scaler):
    """Convert environment state to BC format using SAVED scaler."""
    
    # Extract the 3 features your BC was trained on
    heading_error = env_state[0]  # Adjust based on your environment
    path_distance = env_state[1]
    progress = env_state[2]
    
    bc_state = np.array([heading_error, path_distance, progress])
    
    # Apply the SAVED scaler (only transform, never fit)
    bc_state_normalized = scaler.transform(bc_state.reshape(1, -1))
    
    return bc_state_normalized

def decode_action(action_idx):
    stim_dirs = [-1, 0, 1]
    if action_idx < 12:
        d_idx = action_idx // 4
        f_idx = action_idx % 4
        return (stim_dirs[d_idx], f_idx)
    else:
        return (None, None)
    

# Load the trained model
policy_net = PolicyNetwork(input_dim=3, output_dim=13, hidden_dims=[128, 64])
checkpoint = torch.load('data/checkpoints/bc_policy.pt', map_location='cpu')
policy_net.load_state_dict(checkpoint['model_state_dict'])
policy_net.eval()

# Load the SAVED scaler (crucial!)
with open('data/checkpoints/state_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
state = env._get_state()
# Your main testing loop
done = False
while not done:
    # Process state using the saved scaler
    bc_state = process_state_for_bc(state, scaler)  # Pass scaler
    
    # Get action
    with torch.no_grad():
        state_tensor = torch.FloatTensor(bc_state).unsqueeze(0)
        output = policy_net(state_tensor)
        action_idx = torch.argmax(output, dim=-1).item()
        print(action_idx)
    action = decode_action(action_idx)
    print(action)
    state, reward, done = env.step(action)
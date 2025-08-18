import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from env.cyborg_env import CyborgInsectEnv

# -- Hyperparameters --
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
MEMORY_SIZE = 20000
TARGET_UPDATE = 500
NUM_EPISODES = 12500
MAX_STEPS = 1250
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 6000000

# -- Model Directory -- 
model_dir = r"G:\biorobotics\data\ClosedLoopControl\RLFramework\models"
checkpoint_path = os.path.join(model_dir, "policy_net_ep12500.pth")
os.makedirs(model_dir, exist_ok=True)


#Image height and lengths
image_height = 800
image_length = 1200

# Generate parameter t for one full sine cycle
t = np.linspace(0, 2 * np.pi, 1000)

# x runs from 50 to image_length - 50
x = 100 + (image_length - 150) * t / (2 * np.pi)

# y is a sine wave with amplitude image_height
y = 400 + (image_height/2 - 100) * np.sin(t)
path = np.column_stack((x, y))
path_int = np.round(path).astype(np.int32)


env = CyborgInsectEnv(
    path=path_int  # Example path
)
n_actions = 3 * 4 + 1  # e.g., 3 directions x 4 frequencies
state_dim = env.reset().shape[0]

# -- Network definition --
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

policy_net = DQN(state_dim, n_actions)
target_net = DQN(state_dim, n_actions)
target_net.load_state_dict(torch.load(checkpoint_path))
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# -- Replay buffer --
memory = deque(maxlen=MEMORY_SIZE)

def select_action(state, steps_done):
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    if random.random() < eps:
        return random.randrange(n_actions)
    else:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return policy_net(s).max(1)[1].item()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*transitions)

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + GAMMA * max_next_q * (1 - dones)
    loss = nn.functional.mse_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def decode_action(action_idx):
    stim_dirs = [-1, 0, 1]
    if action_idx < 12:
        d_idx = action_idx // 4
        f_idx = action_idx % 4
        return (stim_dirs[d_idx], f_idx)
    else:
        return (None, None)


def main():
    # Check if CUDA (GPU support) is available
    print("CUDA Available:", torch.cuda.is_available())

    # Check the number of GPUs available
    print("Number of GPUs:", torch.cuda.device_count())

    # Get the current device ID (typically 0 if you have one GPU)
    print("Current GPU Device:", torch.cuda.current_device())

    # Get the device name of your GPU
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Check where a tensor is located (should say 'cuda:0' if on GPU)
    x = torch.rand(3, 3).to('cuda')
    print("Tensor device:", x.device)
    # -- Training loop --
    steps_done = 0
    ep_start = 12500
    for episode in range(ep_start, ep_start + NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        for t in range(MAX_STEPS):
            action_idx = select_action(state, steps_done)
            action = decode_action(action_idx)
            next_state, reward, done = env.step(action)
            # if t % 50 == 0: 
            #     env.render(show=False)
            memory.append((state, action_idx, reward, next_state, float(done)))
            state = next_state
            total_reward += reward
            steps_done += 1
            optimize_model()
            if steps_done % TARGET_UPDATE == 0:  
                target_net.load_state_dict(policy_net.state_dict())
            if done:
                break
        print(f"Episode {episode+1}: Total reward = {total_reward:.2f}")
        # Optionally save model every N episodes
        if (episode+1) % 100 == 0:
            model_path = os.path.join(model_dir, f"policy_net_ep{episode+1}.pth")
            torch.save(policy_net.state_dict(), model_path)
    print("Training finished.")

    # Save final model
    torch.save(policy_net.state_dict(), r"G:\biorobotics\data\ClosedLoopControl\RLFramework\models\policy_net_final.pth")

if __name__ == "__main__": 
    main()
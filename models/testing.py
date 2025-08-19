import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Load your environment and network ---
from env.cyborg_env import CyborgInsectEnv   # Update with your actual import
from training.train import *

# Set up the environment
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
# Load the trained model
input_dim = env.reset().shape[0]    # or explicitly e.g. 6
n_actions = 13                      # as defined before

policy_net = DQN(input_dim, n_actions)

policy_net.load_state_dict(torch.load('models\policy_net_ep23900.pth'))
policy_net.eval()


state = env.reset()

# Tracking for plotting
positions = [env.position.copy()]
rewards = []
progresses = [env.prev_progress]

done = False

plt.ion()  # Interactive plotting on
i = 0
# --- Main Greedy Policy Rollout ---
while not done:
    i += 1
    # Select best (greedy) action from policy
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_idx = policy_net(s).max(1)[1].item()

    action = decode_action(action_idx)
    state, reward, done = env.step(action)

    positions.append(env.position.copy())
    rewards.append(reward)
    progresses.append(env.prev_progress)

    # Live plot: animate
    if i % 20 == 0:
        env.render(show=False)  # update the figure without blocking

print(reward)
plt.ioff()  # Turn off interactive mode

# --- Final static plot of run ---
positions = np.array(positions)
plt.figure()
plt.plot(env.path[:, 0], env.path[:, 1], '--', label='Path')
plt.plot(positions[:, 0], positions[:, 1], '-', label='Agent Trajectory')
plt.scatter(positions[0, 0], positions[0, 1], c='green', label='Start')
plt.scatter(positions[-1, 0], positions[-1, 1], c='red', label='End')
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Full greedy policy rollout")
plt.axis('equal')
plt.show()
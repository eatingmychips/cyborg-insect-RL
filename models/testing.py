import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Load your environment and network ---
from env.cyborg_env import CyborgInsectEnv   # Update with your actual import
from training.train import *



env = CyborgInsectEnv(

)
# Load the trained model
input_dim = env.reset().shape[0]    # or explicitly e.g. 6
n_actions = 11                 

policy_net = DQN(input_dim, n_actions)

policy_net.load_state_dict(torch.load('models\policy_net_final.pth'))
policy_net.eval()


state = env.reset()

# Tracking for plotting
positions = [env.position.copy()]
rewards = []

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

    print(action)
    positions.append(env.position.copy())
    rewards.append(reward)
    
    # Live plot: animate
    if i % 1 == 0:
        env.render(show=True)  # update the figure without blocking


print('Time Steps', i)
plt.ioff()  # Turn off interactive mode

# --- Final static plot of run ---
positions = np.array(positions)
plt.figure()
plt.plot(positions[:, 0], positions[:, 1], '-', label='Agent Trajectory')
plt.scatter(positions[0, 0], positions[0, 1], c='green', label='Start')
plt.scatter(positions[-1, 0], positions[-1, 1], c='red', label='End')
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Full greedy policy rollout")
plt.axis('equal')
plt.show()
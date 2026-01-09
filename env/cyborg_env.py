import numpy as np
import matplotlib.pyplot as plt
import random 


REWARD_CONSTANT = 0.1
STEP_PENALTY = 0.1


class CyborgInsectEnv:
    def __init__(self, stim_freqs=[10, 20, 30, 40], obstacles = None):
        self.stim_freqs = stim_freqs
        self.obstacles = obstacles

        # OU Dispersion for heading
        self.ou_theta = 0.3  # speed of mean reversion
        self.ou_mu = 0         # desired mean (0 rad)
        self.ou_sigma = np.deg2rad(3)  # volatility (typical drift in rad)
        self.ou_drift = 0      # running drift
        
        # OU Dispersion for velocity 
        self.ou_theta_v = 1
        self.ou_mu_v = 0.5
        self.ou_sigma_v = 0.1
        self.velocity = 0.5
        

        self.x_min = -100
        self.x_max = 100
        self.y_min = 25
        self.y_max = 100
        
        self.target = self._sample_random_target()
        self.heading = 0
        self.step_no = 0
        self.max_steps = 500
        self.position = np.array([0, 0])
        self.done = False
        self.progress = 0
        self.action_cost = 5
        self.heading_drift_list = np.deg2rad([-2, -1, 0, 0, 1, 2])
        self.state = self._get_state()
        self.reset()
        
    def reset(self):
        self.step_no = 0
        self.heading = np.deg2rad(np.random.normal(loc=90, scale=45))
        self.progress = 0
        self.position = [0, 0]
        self.done = False
        self.target = self._sample_random_target()
        return self._get_state()

    def _sample_random_target(self):
        # e.g., uniform sample within bounds
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        return np.array([x, y])
    

    def _angle_wrap(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle <= -np.pi:
            angle += 2 * np.pi
        return angle


    def step(self, action):
        """RL step loop with optional cooldown and action penalty."""
        reward = 0.0

        # Unpack action
        stim_direction, freq_idx = action


        self.ou_drift += self.ou_theta * (self.ou_mu - self.ou_drift) + self.ou_sigma * np.random.normal()
        
        self.velocity += self.ou_theta_v *(self.ou_mu_v - self.velocity) + self.ou_sigma_v * np.random.normal()
        
        self.heading = self._angle_wrap(self.heading + self.ou_drift)

        if stim_direction is not None:
            freq = self.stim_freqs[freq_idx]
            heading_change = np.radians(stim_direction * freq * 1.5)
            self.heading = self._angle_wrap(self.heading + heading_change)
            reward -= self.action_cost
            if stim_direction == 0: 
                boost = 2 / (1 + 10 * self.velocity) * (freq_idx + 1) * 1.5
                self.velocity += boost
                reward -= 2 * self.action_cost

        self.position += np.array([np.cos(self.heading), np.sin(self.heading)]) * self.velocity
        prev_progress = self.state[1]


        # Compute state (same variables used for reward)
        self.state = self._get_state()
        
        angle_diff = self.state[0]
        progress = self.state[1]

        reward += (prev_progress - progress - angle_diff * 0.5)
        if progress <= 10: 
            self.done = True 
            reward += 500

        return self.state, reward, self.done

    
    def _get_state(self):
        """Returns the observation vector for the agent."""
        goal_dist = np.linalg.norm(self.target - self.position)
        
        direction = np.array(self.target) - np.array(self.position)
        angle_to_tangent = np.arctan2(direction[1], direction[0])
        angle_diff = angle_to_tangent - self.heading
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Normalise the heading difference between -pi and pi 
        angle_diff_norm = angle_diff / np.pi

        # Base observation
        state_components = [
            angle_diff_norm,
            goal_dist   # Setup so 0 is finishing at goal 
        ]

        return np.array(state_components, dtype=np.float32)

    def render(self, agent_color='blue', target_color='red', show=True):
        plt.clf()
 
        # Plot the current position of the agent
        plt.scatter(self.position[0], self.position[1], color=agent_color, label='Agent')
    
        # Plot target point
        target_point = self.target
        plt.scatter(target_point[0], target_point[1], color=target_color, label='Target', marker='x')
    

        # Add an arrow for heading
        arrow_length = 5 # Set arrow length as desired (adjust if units are small/large)
        dx = arrow_length * np.cos(self.heading)
        dy = arrow_length * np.sin(self.heading)
        plt.arrow(self.position[0], self.position[1], dx, dy,
                head_width=arrow_length*0.3, head_length=arrow_length*0.3,
                fc=agent_color, ec=agent_color, linewidth=2, label='Heading')

        
        plt.legend(loc="best")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.title("Cyborg Insect Environment")
        plt.pause(0.01)
        if show:
            plt.show(block=False)
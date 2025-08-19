import numpy as np
import matplotlib.pyplot as plt


class CyborgInsectEnv:
    def __init__(self, path, lookahead=100, stim_freqs=[10, 20, 30, 40], time_step=0.05,
                 baseline_velocity = 2):
        self.path = np.array(path, dtype=np.float32)
        self.lookahead = lookahead
        self.stim_freqs = stim_freqs
        self.time_step = time_step
        self.baseline_velocity = baseline_velocity       # e.g., 0.01 units per step
        self.active_controlled_velocity = 0.0
       
        self.controlled_until = 0.0
        self.total_arc_length = np.sum(np.linalg.norm(self.path[1:] - self.path[:-1], axis=1))
        self.x_min = 0
        self.x_max = 1200  # Set by your image_length
        self.y_min = 0
        self.y_max = 800   # Set by your image_height

        self.include_history = False     # Start simple
        self.use_cooldown = False        # Start in penalty-only mode
        self.action_cost = 0.1           # Action penalty value
        self.stim_min_interval = 0.5     # Cooldown duration for when enabled
        self.time_since_last_stim = 0.0
        
        self.reset()
        
    def reset(self):
        self.position = np.array(self.path[0], dtype=np.float32)
        self.heading = 0.75 # radians
        self.stim_history = []  # for habituation modeling
        self.done = False
        self.sim_time = 0.0
        self.controlled_until = 0.0
        self.prev_progress = self._project_position_onto_path(self.position)
        self.visited_waypoints = set()
        return self._get_state()

    def _angle_wrap(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle <= -np.pi:
            angle += 2 * np.pi
        return angle

    def find_target_point(self):
        distances = np.linalg.norm(self.path - self.position, axis=1)
        closest_idx = np.argmin(distances)
        # Choose a target index lookahead_steps ahead (integer), or use arc length as advanced option
        lookahead_steps = int(self.lookahead)  # set this as an attribute
        target_idx = min(closest_idx + lookahead_steps, len(self.path) - 1)
        return self.path[target_idx]


    def step(self, action):
        """RL step loop with optional cooldown and action penalty."""
        reward = 0.0
        self.sim_time += self.time_step

        # Unpack action
        stim_direction, freq_idx = action

        # Always apply baseline move
        self.heading = self._angle_wrap(self.heading)
        self.position += np.array([np.cos(self.heading), np.sin(self.heading)]) * self.baseline_velocity

        # Cooldown logic
        can_stim = True
        if self.use_cooldown:
            can_stim = self.time_since_last_stim >= self.stim_min_interval

        # Apply stimulation if allowed
        if stim_direction is not None and can_stim:
            freq = self.stim_freqs[freq_idx]
            heading_change = np.radians(stim_direction * freq * 1.5)
            self.heading = self._angle_wrap(self.heading + heading_change)

            # Forward burst if stim_direction == 0
            self.active_controlled_velocity = freq * 0.05 if stim_direction == 0 else 0.0

            # Track for history
            self.last_stim_freq = freq_idx
            self.stim_history.append(stim_direction)
            if self.use_cooldown:
                self.time_since_last_stim = 0.0

        # Update cooldown timer if used
        if self.use_cooldown:
            self.time_since_last_stim += self.time_step

        # Apply controlled velocity (one-step burst in penalty-only mode)
        self.position += np.array([np.cos(self.heading), np.sin(self.heading)]) * self.active_controlled_velocity
        self.active_controlled_velocity = 0.0

        # Action penalty for sparsity
        if stim_direction is not None:
            reward -= self.action_cost

        # Compute state (same variables used for reward)
        state = self._get_state(include_history=self.include_history)
        # Find target point for heading delta
        target_point = self.find_target_point()
        path_index = np.argmin(np.linalg.norm(self.path - target_point, axis=1))

        # Compute tangent vector
        if path_index == 0:
            tangent_vector = self.path[1] - self.path[0]
        elif path_index < len(self.path) - 1:
            tangent_vector = self.path[path_index + 1] - self.path[path_index]
        else:
            tangent_vector = self.path[path_index] - self.path[path_index - 1]

        tangent_vector = tangent_vector.astype(np.float64)
        tangent_vector /= np.linalg.norm(tangent_vector)

        # Tangent angle
        angle_to_tangent = np.arctan2(tangent_vector[1], tangent_vector[0])

        # Heading delta
        heading_delta = self._angle_wrap(angle_to_tangent - self.heading)

        min_distance = state[1]
        progress_along_path = state[2]

        # --- Reward shaping ---
        # Progress reward
        progress_delta = progress_along_path - self.prev_progress
        reward += progress_delta * 100.0
        self.prev_progress = progress_along_path

        # Off-path handling
        max_deviation = 200
        if min_distance > max_deviation:
            self.done = True
            reward -= 250
        else:
            reward -= 5.0 * (min_distance / max_deviation) ** 2

        # Heading alignment penalty
        heading_penalty = (abs(heading_delta) / np.pi) ** 2
        reward -= heading_penalty * 5.0

        # Goal reward
        if np.linalg.norm(self.position - self.path[-1]) <= 125:
            self.done = True
            reward += 500

        return state, reward, self.done


    
    def discretize_heading_delta(self, heading_delta, num_bins=18):
        bin_size = 2 * np.pi / num_bins
        shifted = heading_delta + np.pi  # Shift to [0, 2π] range
        bin_index = int(shifted // bin_size)
        bin_index = min(bin_index, num_bins - 1)  # Clip to max bin
        return bin_index

    
    def _get_state(self):
        """Returns the observation vector for the agent."""

        # Distances
        min_distance = np.min(np.linalg.norm(self.path - self.position, axis=1))
        progress_along_path = self._project_position_onto_path(self.position)  # 0.0–1.0

        # Path tangent
        target_point = self.find_target_point()
        path_index = np.argmin(np.linalg.norm(self.path - target_point, axis=1))
        if path_index == 0:
            tangent_vector = self.path[1] - self.path[0]
        elif path_index < len(self.path) - 1:
            tangent_vector = self.path[path_index + 1] - self.path[path_index]
        else:
            tangent_vector = self.path[path_index] - self.path[path_index - 1]
        tangent_vector /= np.linalg.norm(tangent_vector)

        angle_to_tangent = np.arctan2(tangent_vector[1], tangent_vector[0])
        heading_delta = self._angle_wrap(angle_to_tangent - self.heading)

        # Discretise heading
        disc_heading = self.discretize_heading_delta(heading_delta)

        # Base observation
        state_components = [
            disc_heading,
            min_distance,
            progress_along_path,
        ]

        return np.array(state_components, dtype=np.float32)

    def _project_position_onto_path(self, position):
        # Find the segment of the path closest to the agent
        distances = np.linalg.norm(self.path - position, axis=1)
        closest_idx = np.argmin(distances)
        
        if closest_idx == len(self.path) - 1:
            # At or past the end
            arc_length = np.sum(np.linalg.norm(self.path[1:] - self.path[:-1], axis=1))
            return 100.0  # End of path

        # Compute projection onto the segment [closest_idx, closest_idx+1]
        p1 = self.path[closest_idx]
        p2 = self.path[closest_idx + 1]
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len == 0:
            lambda_proj = 0
        else:
            lambda_proj = np.dot(position - p1, seg_vec) / seg_len**2
            lambda_proj = np.clip(lambda_proj, 0, 1)
        # Actual projected point
        proj_point = p1 + lambda_proj * seg_vec
        
        if closest_idx == 0:
            arc_before = 0.0
        else:
            arc_before = np.sum(np.linalg.norm(self.path[1:closest_idx+1] - self.path[:closest_idx], axis=1))
        arc_here = np.linalg.norm(proj_point - p1)
        arc_length = arc_before + arc_here

    # Return progress as a percentage (1 to 100)
        return 100.0 * arc_length / self.total_arc_length

    def render(self, path_color='gray', agent_color='blue', target_color='red', show=True):
        plt.clf()
        # Plot the path
        plt.plot(self.path[:, 0], self.path[:, 1], color=path_color, linestyle="--", label='Path')
        
        # Plot the current position of the agent
        plt.scatter(self.position[0], self.position[1], color=agent_color, label='Agent')
        
        plt.scatter(self.path[-1][0], self.path[-1][1])
        # Plot target point
        target_point = self.find_target_point()
        plt.scatter(target_point[0], target_point[1], color=target_color, label='Target', marker='x')
        
        waypoints = self.path[::150]
        for point in waypoints:
            plt.scatter(point[0], point[1])
        # Add an arrow for heading
        arrow_length = 30 # Set arrow length as desired (adjust if units are small/large)
        dx = arrow_length * np.cos(self.heading)
        dy = arrow_length * np.sin(self.heading)
        plt.arrow(self.position[0], self.position[1], dx, dy,
                head_width=arrow_length*0.3, head_length=arrow_length*0.3,
                fc=agent_color, ec=agent_color, linewidth=0.5, label='Heading')

        
        plt.axis('equal')
        plt.legend(loc="best")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Cyborg Insect Environment")
        plt.pause(0.01)
        if show:
            plt.show(block=False)
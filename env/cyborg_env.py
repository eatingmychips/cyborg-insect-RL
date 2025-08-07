import numpy as np
import matplotlib.pyplot as plt


class CyborgInsectEnv:
    def __init__(self, path, tangent_control_thresh=0.2, lookahead=200, levels=9, stim_freqs=[10, 20, 30, 40], time_step=0.05, stim_min_interval = 1., 
                 baseline_velocity = 0.01, baseline_angle_std = 0.01):
        self.path = path
        self.tangent_control_thresh = tangent_control_thresh
        self.lookahead = lookahead
        self.levels = levels
        self.stim_freqs = stim_freqs
        self.time_step = time_step
        self.stim_min_interval = stim_min_interval
        self.time_since_last_stim = 0.0
        self.baseline_velocity = baseline_velocity       # e.g., 0.01 units per step
        self.baseline_angle_std = baseline_angle_std
        self.controlled_until = 0.0
        self.total_arc_length = np.sum(np.linalg.norm(self.path[1:] - self.path[:-1], axis=1))
        self.reset()
        
    def reset(self):
        self.position = np.array(self.path[0], dtype=np.float32)
        self.heading = 0.0  # radians
        self.stim_history = []  # for habituation modeling
        self.done = False
        self.sim_time = 0.0
        self.controlled_until = 0.0
        self.prev_progress = self._project_position_onto_path(self.position)

        return self._get_state()

    def _angle_wrap(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle <= -np.pi:
            angle += 2 * np.pi
        return angle

    def find_target_point(self):
        distances = np.linalg.norm(self.path - self.position, axis=1)
        for i, dist in enumerate(distances):
            if dist > self.lookahead:
                return self.path[i]
        return self.path[-1]

    def step(self, action):
        self.sim_time += self.time_step  # Update simulation time
        # Baseline heading drift and forward velocity -- as in previous answer
        heading_drift = np.random.normal(0, self.baseline_angle_std)
        self.heading += heading_drift
        self.heading = self._angle_wrap(self.heading)
        self.position += np.array([np.cos(self.heading), np.sin(self.heading)]) * self.baseline_velocity

        self.time_since_last_stim += self.time_step
        stim_direction, freq_idx = action

        # Check if we can stimulate
        if self.time_since_last_stim >= self.stim_min_interval:
            if stim_direction is not None:
                freq = self.stim_freqs[freq_idx]
                habituation_factor = 0.5 if len(self.stim_history) >= 2 and \
                    self.stim_history[-1] == stim_direction and self.stim_history[-2] == stim_direction else 1.0
                heading_change = np.radians(stim_direction * freq * habituation_factor)
                self.heading += heading_change
                self.heading = self._angle_wrap(self.heading)
                
                # Compute and set controlled velocity burst for the next interval
                if stim_direction == 0:
                    self.active_controlled_velocity = freq * 0.05
                else:
                    self.active_controlled_velocity = freq * 0.02

                self.controlled_until = self.sim_time + self.stim_min_interval
                self.time_since_last_stim = 0.0
                self.stim_history.append(stim_direction)
            else:
                # No stimulation, ensure controlled velocity is zeroed until next stim
                self.active_controlled_velocity = 0.0
                self.controlled_until = self.sim_time + self.stim_min_interval
                self.time_since_last_stim = 0.0

        # Now: apply current controlled velocity, but ONLY if within interval
        if self.sim_time < self.controlled_until:
            self.position += np.array([np.cos(self.heading), np.sin(self.heading)]) * self.active_controlled_velocity
        else:
            self.active_controlled_velocity = 0.0  # Clear burst when time is up

        # Trim stim_history, compute reward, return state, etc. (unchanged)

        if len(self.stim_history) > 3:
            self.stim_history.pop(0)

        state = self._get_state()
        distance_to_path = state[3]
        current_progress = self._project_position_onto_path(self.position)
        reward = current_progress - self.prev_progress
        self.prev_progress = current_progress
        target_point = self.find_target_point()
        angle_error = abs(self._angle_wrap(np.arctan2(target_point[1] - self.position[1], target_point[0] - self.position[0]) - self.heading))
        reward = current_progress - angle_error - distance_to_path
        self.done = False
        return self._get_state(), reward, self.done


    def _get_state(self):
        target_point = self.find_target_point()
        distance_to_path = np.linalg.norm(self.path - self.position, axis=1)
        min_distance = distance_to_path.min()
        
        path_index = np.argmin(np.linalg.norm(self.path - target_point, axis=1))
        if path_index == 0:
            tangent_vector = self.path[1] - self.path[0]
        elif path_index < len(self.path) - 1:
            tangent_vector = self.path[path_index + 1] - self.path[path_index]
        else:
            tangent_vector = self.path[path_index] - self.path[path_index - 1]
        tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
        angle_to_tangent = np.arctan2(tangent_vector[1], tangent_vector[0])
        heading_delta = self._angle_wrap(angle_to_tangent - self.heading)
   
        # Example state: [x, y, heading_delta, min_distance, last_stim_dir, last_freq_idx]
        last_stim_dir = self.stim_history[-1] if self.stim_history else 0
        last_freq_idx = 0
        state = np.array([
            self.position[0],
            self.position[1],
            heading_delta,
            min_distance,
            last_stim_dir,
            last_freq_idx
        ], dtype=np.float32)
        return state


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
        
        # Plot target point
        target_point = self.find_target_point()
        plt.scatter(target_point[0], target_point[1], color=target_color, label='Target', marker='x')
        
        # Add an arrow for heading
        arrow_length = 25 # Set arrow length as desired (adjust if units are small/large)
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
import numpy as np

class CyborgInsectEnv:
    def __init__(self, path, tangent_control_thresh=0.2, lookahead=0.3, levels=9, stim_freqs=[10, 20, 30, 40]):
        self.path = path
        self.tangent_control_thresh = tangent_control_thresh
        self.lookahead = lookahead
        self.levels = levels
        self.stim_freqs = stim_freqs
        self.reset()
        
    def reset(self):
        self.position = np.array(self.path[0], dtype=np.float32)
        self.heading = 0.0  # radians
        self.stim_history = []  # for habituation modeling
        self.done = False
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
        # action: (stim_direction, freq_idx)
        stim_direction, freq_idx = action
        freq = self.stim_freqs[freq_idx]
        # Habituation: after 2 repeated stimuli, reduce heading effect
        habituation_factor = 0.7 if len(self.stim_history) >= 2 and \
            self.stim_history[-1] == stim_direction and self.stim_history[-2] == stim_direction else 1.0
        # Update heading based on stimulation + habituation
        heading_change = stim_direction * freq * 0.1 * habituation_factor
        self.heading += heading_change
        self.heading = self._angle_wrap(self.heading)
        # Velocity defined by frequency and direction: straight moves faster
        velocity = freq * 0.05 if stim_direction == 0 else freq * 0.02
        self.position += np.array([np.cos(self.heading), np.sin(self.heading)]) * velocity
        self.stim_history.append(stim_direction)
        if len(self.stim_history) > 3:
            self.stim_history.pop(0)
        # Calculate reward based on distance to lookahead and heading error
        target_point = self.find_target_point()
        distance_to_target = np.linalg.norm(target_point - self.position)
        angle_error = abs(self._angle_wrap(np.arctan2(target_point[1] - self.position[1], target_point[0] - self.position[0]) - self.heading))
        reward = -distance_to_target - 0.5 * angle_error  # dense, always provides feedback
        self.done = False  # Add success/failure conditions as needed
        return self._get_state(), reward, self.done
    
    
# TODO: Think about already avaialable environments 

    def _get_state(self):
        target_point = self.find_target_point()
        distance_to_path = np.linalg.norm(self.path - self.position, axis=1)
        min_distance = distance_to_path.min()
        if min_distance <= self.tangent_control_thresh:
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
        else:
            heading_delta = self._angle_wrap(np.arctan2(target_point[1] - self.position[1], target_point[0] - self.position[0]) - self.heading)
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

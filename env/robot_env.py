import gymnasium as gym
import numpy as np


class RobotEnv(gym.Env):
    def __init__(self, step_size=0.1, turn_angle=0.1, max_steps=500, num_obstacles=8, robot_radius=0.02, target_radius=0.05, target_position=(0.5, 0.5)):
        super(RobotEnv, self).__init__()

        # Параметры среды
        self.step_size = step_size
        self.turn_angle = turn_angle
        self.num_obstacles = num_obstacles
        self.robot_radius = robot_radius
        self.target_radius = target_radius
        self.target_position = target_position
        self.max_dist = np.sqrt(2)


        # Начальные значения (будут перезаписываться в reset())
        self.robot_x = 0
        self.robot_y = 0
        self.robot_angle = 0
        self.robot_speed_x = 0
        self.robot_speed_y = 0
        self.target_x = 0
        self.target_y = 0
        self.obstacles = []
        self.current_step = 0
        
        # Пространство действий
        self.action_space = gym.spaces.Discrete(5)


        # Индексы: 0-1 pos, 2-3 vel, 4 angle, 5 dist, 6 angle_to_target, 7-14 rays
        OBS_LOW  = np.array([
            0, 0,           # x, y
            -1, -1,         # v_x, v_y
            -1,             # angle
            0,              # dist_to_target
            -1,             # angle_to_target
            0, 0, 0, 0, 0, 0, 0, 0,  # rays
        ], dtype=np.float32)

        OBS_HIGH = np.ones(15, dtype=np.float32)
        
        # Пространство наблюдений
        self.observation_space = gym.spaces.Box(
            low=OBS_LOW,
            high=OBS_HIGH,
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        obs = np.empty(15, dtype=np.float32)
        obs[0:2] = self.robot_x, self.robot_y
        obs[2:4] = self.robot_speed_x / self.step_size, self.robot_speed_y / self.step_size
        obs[4] = self.robot_angle / np.pi
        obs[5] = np.linalg.norm(np.array([self.target_x, self.target_y]) - np.array([self.robot_x, self.robot_y])) / self.max_dist
        ...
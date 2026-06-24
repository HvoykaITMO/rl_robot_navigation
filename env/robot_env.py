import gymnasium as gym
import numpy as np
from utils import constants as c


class RobotEnv(gym.Env):
    def __init__(
        self,
        step_size: float,
        turn_angle: float,
        max_steps: int,
        num_obstacles: int,
        robot_radius: float,
        target_radius: float,
    ):
        super(RobotEnv, self).__init__()

        # Параметры среды
        self.step_size = step_size
        self.turn_angle = turn_angle
        self.num_obstacles = num_obstacles
        self.robot_radius = robot_radius
        self.target_radius = target_radius
        self.max_steps = max_steps

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
        self.ray_endpoints = [] # Для визуализаций
        
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

    def _ray_boundary_intersection(self, dir_x, dir_y):  # Пересечение с границами карты
            # Любая точка на луче (параметрически): x = ray_x + t * dir_x ; y = ray_y + t * dir_y (1)
            # Пересечение луча с границой карты [0, 1] x [0, 1] = приравнять координату луча к координате границы
            # Считаем для каждого луча и берем min (ex. Пересеч с гор. прямой y = 0 (нижн. граница)):
            # robot_y + t * dir_y = 0; t = -robot_y / dir_y

            min_t = c.RAY_MAX_DIST
            
            # Проверяем 4 границы
            boundaries = [
                (0, 'y'),  # Нижняя граница: y = 0
                (1, 'y'),  # Верхняя граница: y = 1
                (0, 'x'),  # Левая граница: x = 0
                (1, 'x')   # Правая граница: x = 1
            ]
            
            for value, axis in boundaries:
                if axis == 'y':
                    if abs(dir_y) > 1e-6:  # Избегаем деления на 0
                        t = (value - self.robot_y) / dir_y
                        if 0 < t < min_t:
                            min_t = t
                else:  # axis == 'x'
                    if abs(dir_x) > 1e-6:
                        t = (value - self.robot_x) / dir_x
                        if 0 < t < min_t:
                            min_t = t
            
            return min_t

    def _ray_circle_intersection(self, dir_x, dir_y, obs_x, obs_y, obs_radius):  # Пересечение препятствиями
            # Любая точка на луче (параметрически): x = ray_x + t * dir_x ; y = ray_y + t * dir_y (1)
            # Уравнение окружности: (x - obs_x)**2 + (y - obs_y)**2 = obs_radius**2 (2)
            # Подставляем (1) в (2) и получаем квадратное уравнение и, по дискриминанту, получаем пересечение

            # вектор от начала луча до центра окружности
            fx = self.robot_x - obs_x
            fy = self.robot_y - obs_y
            
            # Коэффициентры квадратного уравнения
            a = dir_x**2 + dir_y**2
            b = 2 * (fx * dir_x + fy * dir_y)
            circle_c = fx**2 + fy**2 - obs_radius**2

            discriminant = b**2 - 4 * a * circle_c

            if discriminant < 0:
                return c.RAY_MAX_DIST # Нет пересечений

            # Корни
            t1 = (-b - np.sqrt(discriminant)) / (2 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2 * a)

            # Берём первое пересечение, если оно не за лучом ( min корень > 0), иначе второй корень
            t = min(t1, t2) if min(t1, t2) > 0 else max(t1, t2)

            if t > 0 and t < c.RAY_MAX_DIST:
                return t
            else:
                return c.RAY_MAX_DIST

    def _get_obs(self) -> np.ndarray:
        
        obs = np.empty(15, dtype=np.float32)
        obs[0:2] = self.robot_x, self.robot_y
        obs[2:4] = self.robot_speed_x / self.step_size, self.robot_speed_y / self.step_size
        obs[4] = self.robot_angle / np.pi
        obs[5] = np.linalg.norm(np.array([self.target_x, self.target_y]) - np.array([self.robot_x, self.robot_y])) / c.RAY_MAX_DIST
        
        # Нормализация относительного угла
        dx, dy = self.target_x - self.robot_x, self.target_y - self.robot_y
        absolute_angle_to_target = np.atan2(dy, dx) # [-pi, pi]
        relative_angle_to_target = absolute_angle_to_target - self.robot_angle
        relative_angle_to_target = (relative_angle_to_target + np.pi) % (2 * np.pi) - np.pi # Может выйти за диапазон [-pi, pi], а так не выйдет
        obs[6] = relative_angle_to_target / np.pi


        # Нормализация расстояний, полученных лучами
        self.ray_endpoints = []  # Для визуализации 
        for i in range(c.RAYS_AMOUNT_GENERATION):
            ray_angle = self.robot_angle + (2 * np.pi * i) / c.RAYS_AMOUNT_GENERATION
            dir_x = np.cos(ray_angle)
            dir_y = np.sin(ray_angle)

            boundary_distance = self._ray_boundary_intersection(
                    dir_x, dir_y
                )
            min_distance = min(boundary_distance, c.RAY_MAX_DIST)
            for obs_x, obs_y, obs_radius in self.obstacles:
                obs_distance = self._ray_circle_intersection(
                    dir_x, dir_y,
                    obs_x, obs_y, obs_radius,
                )
                min_distance = min(min_distance, obs_distance)


            # Для визуализаций
            end_x = self.robot_x + min_distance * dir_x
            end_y = self.robot_y + min_distance * dir_y
            self.ray_endpoints.append((end_x, end_y))

            obs[7+i] = min_distance / c.RAY_MAX_DIST
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_x = self.np_random.uniform(self.robot_radius + c.MAP_EDGES_BUFFER_DURING_GENERATION, 1 - self.robot_radius)
        self.robot_y = self.np_random.uniform(self.robot_radius + c.MAP_EDGES_BUFFER_DURING_GENERATION, 1 - self.robot_radius)
        self.robot_speed_x = 0
        self.robot_speed_y = 0
        self.robot_angle = self.np_random.uniform(-np.pi, np.pi)
        self.current_step = 0

        self.obstacles = []
        for i in range(self.num_obstacles):
            is_valid = False
            attempts = 0
            while not is_valid:
                attempts += 1

                if attempts > c.MAX_ATTEMPTS_FOR_WHILE_GENERATION:
                    raise RuntimeError(f"Too many attempts to generate obstacle {i}")

                obs_radius = self.np_random.uniform(c.MIN_OBS_RADIUS_GENERATION, c.MAX_OBS_RADIUS_GENERATION)
                obs_x = self.np_random.uniform(obs_radius + c.MAP_EDGES_BUFFER_DURING_GENERATION, 1 - obs_radius)
                obs_y = self.np_random.uniform(obs_radius + c.MAP_EDGES_BUFFER_DURING_GENERATION, 1 - obs_radius)
                
                is_valid = True

                if np.linalg.norm(np.array([obs_x, obs_y]) - np.array([self.robot_x, self.robot_y])) < self.robot_radius + obs_radius + c.MIN_FROM_ROBOT_TO_OBS_DIST_GENERATION:
                    is_valid = False

                if any((np.linalg.norm(np.array([obs_x, obs_y]) - np.array([other_obs_x, other_obs_y])) < obs_radius + other_obs_radius + c.MIN_FROM_OBS_TO_OBS_DIST_GENERATION for other_obs_x, other_obs_y, other_obs_radius in self.obstacles)):
                    is_valid = False

            self.obstacles.append((obs_x, obs_y, obs_radius))

        is_valid = False
        attempts = 0
        while not is_valid:
            attempts += 1

            if attempts > c.MAX_ATTEMPTS_FOR_WHILE_GENERATION:
                raise RuntimeError(f"Too many attempts to generate target")

            target_x = self.np_random.uniform(self.target_radius + c.MAP_EDGES_BUFFER_DURING_GENERATION, 1 - self.target_radius)
            target_y = self.np_random.uniform(self.target_radius + c.MAP_EDGES_BUFFER_DURING_GENERATION, 1 - self.target_radius)

            is_valid = True

            if np.linalg.norm(np.array([target_x, target_y]) - np.array([self.robot_x, self.robot_y])) < self.robot_radius + self.target_radius + c.MIN_FROM_ROBOT_TO_TARGET_DIST_GENERATION:
                is_valid = False

            if any((np.linalg.norm(np.array([target_x, target_y]) - np.array([obs_x, obs_y])) < obs_radius + self.target_radius + c.MIN_FROM_OBS_TO_TARGET_DIST_GENERATION for obs_x, obs_y, obs_radius in self.obstacles)):
                is_valid = False

        self.target_x = target_x
        self.target_y = target_y

        info = {
            "distance_to_target": float(np.linalg.norm(
                    np.array([self.target_x, self.target_y])
                    - np.array([self.robot_x, self.robot_y]))),
            "is_success": False,
            "crashed": False,
            "steps_taken": self.current_step,
        }

        return self._get_obs(), info

    def step(self, action):
        old_distance = np.linalg.norm(np.array([self.robot_x, self.robot_y]) - np.array([self.target_x, self.target_y]))

        match action:
            case 0: # Стоять
                self.robot_speed_x = 0
                self.robot_speed_y = 0
            case 1: # Вперёд
                self.robot_speed_x = self.step_size * np.cos(self.robot_angle)
                self.robot_speed_y = self.step_size * np.sin(self.robot_angle)
            case 2: # Назад
                self.robot_speed_x = -self.step_size * np.cos(self.robot_angle)
                self.robot_speed_y = -self.step_size * np.sin(self.robot_angle)
            case 3: # Влево
                self.robot_angle += self.turn_angle
                self.robot_angle = (self.robot_angle + np.pi) % (2 * np.pi) - np.pi
                self.robot_speed_x = 0
                self.robot_speed_y = 0
            case 4: # Вправо
                self.robot_angle -= self.turn_angle
                self.robot_angle = (self.robot_angle + np.pi) % (2 * np.pi) - np.pi
                self.robot_speed_x = 0
                self.robot_speed_y = 0
            case _:
                pass

        self.robot_x += self.robot_speed_x
        self.robot_y += self.robot_speed_y

        # Проверка на столкновение с препятствиями или вылет с карты
        crashed = False
        if (self.robot_x <= self.robot_radius + c.FROM_ROBOT_TO_EDGES_DIST_ACCURACY_REGISTRATION 
            or self.robot_x >= 1 - self.robot_radius - c.FROM_ROBOT_TO_EDGES_DIST_ACCURACY_REGISTRATION
            or self.robot_y <= self.robot_radius + c.FROM_ROBOT_TO_EDGES_DIST_ACCURACY_REGISTRATION 
            or self.robot_y >= 1 - self.robot_radius - c.FROM_ROBOT_TO_EDGES_DIST_ACCURACY_REGISTRATION):
            crashed = True
        else:
            for obs_x, obs_y, obs_radius in self.obstacles:
                distance = np.linalg.norm(np.array([obs_x, obs_y]) - np.array([self.robot_x, self.robot_y]))
                if distance <= self.robot_radius + obs_radius + c.FROM_ROBOT_TO_OBS_DIST_ACCURACY_REGISTRATION:
                    crashed = True
                    break
        
        # Проверка, достигли ли цели
        new_distance = np.linalg.norm(np.array([self.robot_x, self.robot_y]) - np.array([self.target_x, self.target_y]))
        reached_target = new_distance < c.FROM_ROBOT_TO_TARGET_DIST_ACCURACY_REGISTRATION

        # Награда
        reward = 0
        distance_diff = old_distance - new_distance
        if distance_diff > 0:
            reward += distance_diff * c.DENSE_REWARD_COEFF
        else:
            reward -= c.TIME_PENALTY
        if reached_target:
            reward += c.LARGE_REWARD
        else:
            reward -= c.LARGE_PENALTY

        self.current_step += 1
        terminated = reached_target or crashed
        truncated = self.current_step >= self.max_steps

        info = {
            "distance_to_target": new_distance,
            "is_success": reached_target,
            "crashed": crashed,
            "steps_taken": self.current_step
        }

        return self._get_obs(), reward, terminated, truncated, info

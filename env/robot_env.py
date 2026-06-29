import gymnasium as gym
import numpy as np
from utils import constants as c


class RobotEnv(gym.Env):
    def __init__(
        self,
        max_steps: int,
        num_obstacles: int,
        robot_radius: float,
        target_radius: float,
    ):
        super(RobotEnv, self).__init__()

        # Параметры среды
        self.num_obstacles = num_obstacles
        self.robot_radius = robot_radius
        self.target_radius = target_radius
        self.max_steps = max_steps

        # Начальные значения (будут перезаписываться в reset())
        self.robot_x = 0
        self.robot_y = 0
        self.robot_angle = 0
        self.robot_speed = 0  # Скалярная скорость робота
        self.current_steering = 0.0  # Сглаженное состояние руления
        self.robot_speed_x = 0 
        self.robot_speed_y = 0
        self.target_x = 0
        self.target_y = 0
        self.obstacles = []
        self.current_step = 0
        self.ray_endpoints = [] # Для визуализаций
        self.previous_action = c.ACTION_STAND # Действие прошлого шага
        self.prev_distance_to_target = 0.0  # прошлая дистанция до цели
        self.prev_min_ray_distance = c.RAY_MAX_DIST  # прошлое минимальное расстояние по лучам
        self.delta_distance_to_target = 0.0  # изменение дистанции
        self.delta_min_ray_distance = 0.0  # изменение дистанции до ближайшей опасности
        
        # Пространство действий
        self.action_space = gym.spaces.Discrete(len(c.DISCRETE_ACTIONS))


        base_low = np.array([
            0.0,   # distance_to_target
            -1.0,  # sin_angle_to_target
            -1.0,  # cos_angle_to_target
            0.0,   # min_ray_distance
            0.0,   # front_ray_distance
            -1.0,  # delta_distance_to_target
            -1.0,  # delta_min_ray_distance
            -1.0,  # closest_ray_sin
            -1.0,  # closest_ray_cos
            0.0,   # target_visible
            -1.0,  # normalized_robot_speed
            -1.0,  # previous_throttle
            -1.0,  # current_steering
            0.0,   # previous_brake
        ], dtype=np.float32)

        OBS_LOW = np.concatenate([
            base_low,
            np.zeros(c.RAYS_AMOUNT_GENERATION, dtype=np.float32),
        ])

        OBS_HIGH = np.ones_like(OBS_LOW, dtype=np.float32)
        
        # Пространство наблюдений
        self.observation_space = gym.spaces.Box(
            low=OBS_LOW,
            high=OBS_HIGH,
            dtype=np.float32,
        )

    def _distance_to_target(self) -> float:
        return np.linalg.norm(
            np.array([self.target_x, self.target_y])
            - np.array([self.robot_x, self.robot_y])
        )

    def _relative_angle_to_target(self) -> float:
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        absolute_angle_to_target = np.atan2(dy, dx)
        relative_angle_to_target = absolute_angle_to_target - self.robot_angle
        return (relative_angle_to_target + np.pi) % (2 * np.pi) - np.pi

    def _get_ray_distances(self) -> np.ndarray:
        ray_distances = np.empty(c.RAYS_AMOUNT_GENERATION, dtype=np.float32)
        self.ray_endpoints = []
        for i in range(c.RAYS_AMOUNT_GENERATION):
            ray_angle = self.robot_angle + (2 * np.pi * i) / c.RAYS_AMOUNT_GENERATION
            dir_x = np.cos(ray_angle)
            dir_y = np.sin(ray_angle)
            boundary_distance = self._ray_boundary_intersection(dir_x, dir_y)
            min_distance = min(boundary_distance, c.RAY_MAX_DIST)
            for obs_x, obs_y, obs_radius in self.obstacles:
                obs_distance = self._ray_circle_intersection(
                    dir_x,
                    dir_y,
                    obs_x,
                    obs_y,
                    obs_radius + self.robot_radius + c.FROM_ROBOT_TO_OBS_DIST_ACCURACY_REGISTRATION,
                )
                min_distance = min(min_distance, obs_distance)
            end_x = self.robot_x + min_distance * dir_x
            end_y = self.robot_y + min_distance * dir_y
            self.ray_endpoints.append((end_x + self.robot_radius * dir_x, end_y + self.robot_radius * dir_y))
            ray_distances[i] = min_distance
        return ray_distances

    def _ray_boundary_intersection(self, dir_x, dir_y):  # Пересечение с границами карты
            # Любая точка на луче (параметрически): x = ray_x + t * dir_x ; y = ray_y + t * dir_y (1)
            # Пересечение луча с границой карты [0, 1] x [0, 1] = приравнять координату луча к координате границы
            # Считаем для каждого луча и берем min (ex. Пересеч с гор. прямой y = 0 (нижн. граница)):
            # robot_y + t * dir_y = 0; t = -robot_y / dir_y

            min_t = c.RAY_MAX_DIST
            
            edge_min = self.robot_radius + c.FROM_ROBOT_TO_EDGES_DIST_ACCURACY_REGISTRATION
            edge_max = 1 - self.robot_radius - c.FROM_ROBOT_TO_EDGES_DIST_ACCURACY_REGISTRATION

            # Проверяем 4 границы безопасной зоны для центра робота
            boundaries = [
                (edge_min, 'y'),  # Нижняя граница
                (edge_max, 'y'),  # Верхняя граница
                (edge_min, 'x'),  # Левая граница
                (edge_max, 'x')   # Правая граница
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

    def _is_target_visible(self) -> float:
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        distance_to_target = np.hypot(dx, dy)

        if distance_to_target < 1e-8:
            return 1.0

        dir_x = dx / distance_to_target
        dir_y = dy / distance_to_target

        visible_distance = self._ray_boundary_intersection(dir_x, dir_y)

        for obs_x, obs_y, obs_radius in self.obstacles:
            obs_distance = self._ray_circle_intersection(
                dir_x,
                dir_y,
                obs_x,
                obs_y,
                obs_radius + self.robot_radius + c.FROM_ROBOT_TO_OBS_DIST_ACCURACY_REGISTRATION,
            )
            visible_distance = min(visible_distance, obs_distance)

        target_reach_distance = max(0.0, distance_to_target - self.target_radius - self.robot_radius)

        return float(visible_distance >= target_reach_distance)

    def _get_obs(self) -> np.ndarray:
        ray_start = 14
        obs_size = ray_start + c.RAYS_AMOUNT_GENERATION

        obs = np.empty(obs_size, dtype=np.float32)

        distance_to_target = self._distance_to_target()
        relative_angle_to_target = self._relative_angle_to_target()
        ray_distances = self._get_ray_distances()

        normalized_distance = distance_to_target / c.RAY_MAX_DIST
        normalized_rays = ray_distances / c.RAY_MAX_DIST

        min_ray_distance = ray_distances.min()
        normalized_min_ray_distance = min_ray_distance / c.RAY_MAX_DIST

        front_ray_distance = ray_distances[0]
        normalized_front_ray_distance = front_ray_distance / c.RAY_MAX_DIST

        closest_ray_index = int(np.argmin(ray_distances))
        closest_ray_relative_angle = (2 * np.pi * closest_ray_index) / c.RAYS_AMOUNT_GENERATION

        if self.robot_speed >= 0.0:
            normalized_robot_speed = self.robot_speed / c.ENV_MAX_FORWARD_SPEED
        else:
            normalized_robot_speed = self.robot_speed / c.ENV_MAX_BACKWARD_SPEED

        previous_action_config = c.DISCRETE_ACTIONS[int(self.previous_action)]

        obs[0] = normalized_distance
        obs[1] = np.sin(relative_angle_to_target)
        obs[2] = np.cos(relative_angle_to_target)
        obs[3] = normalized_min_ray_distance
        obs[4] = normalized_front_ray_distance
        obs[5] = np.clip(self.delta_distance_to_target / c.RAY_MAX_DIST, -1.0, 1.0)
        obs[6] = np.clip(self.delta_min_ray_distance / c.RAY_MAX_DIST, -1.0, 1.0)
        obs[7] = np.sin(closest_ray_relative_angle)
        obs[8] = np.cos(closest_ray_relative_angle)
        obs[9] = self._is_target_visible()
        obs[10] = np.clip(normalized_robot_speed, -1.0, 1.0)
        obs[11] = previous_action_config["throttle"]
        obs[12] = self.current_steering
        obs[13] = previous_action_config["brake"]
        obs[ray_start:obs_size] = normalized_rays

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.obstacles = []
        for i in range(self.num_obstacles):
            is_valid = False
            attempts = 0
            while not is_valid:
                attempts += 1

                if attempts > c.MAX_ATTEMPTS_FOR_WHILE_GENERATION:
                    raise RuntimeError(f"Too many attempts to generate obstacle {i}")

                obs_radius = self.np_random.uniform(c.MIN_OBS_RADIUS_GENERATION, c.MAX_OBS_RADIUS_GENERATION)
                obs_x = self.np_random.uniform(obs_radius + c.MAP_EDGES_BUFFER_DURING_OBS_GENERATION, 1 - obs_radius)
                obs_y = self.np_random.uniform(obs_radius + c.MAP_EDGES_BUFFER_DURING_OBS_GENERATION, 1 - obs_radius)
                
                is_valid = True

                if any((np.linalg.norm(np.array([obs_x, obs_y]) - np.array([other_obs_x, other_obs_y])) < 
                    obs_radius + other_obs_radius + c.MIN_FROM_OBS_TO_OBS_DIST_GENERATION for other_obs_x, other_obs_y, other_obs_radius in self.obstacles)):
                    is_valid = False

            self.obstacles.append((obs_x, obs_y, obs_radius))

        is_valid = False
        attempts = 0
        while not is_valid:
            attempts += 1

            if attempts > c.MAX_ATTEMPTS_FOR_WHILE_GENERATION:
                raise RuntimeError("Too many attempts to generate robot")

            self.robot_x = self.np_random.uniform(self.robot_radius + c.MAP_EDGES_BUFFER_DURING_ROBOT_GENERATION, 1 - self.robot_radius - c.MAP_EDGES_BUFFER_DURING_ROBOT_GENERATION)
            self.robot_y = self.np_random.uniform(self.robot_radius + c.MAP_EDGES_BUFFER_DURING_ROBOT_GENERATION, 1 - self.robot_radius - c.MAP_EDGES_BUFFER_DURING_ROBOT_GENERATION)
            
            is_valid = True

            if any((np.linalg.norm(np.array([self.robot_x, self.robot_y]) - np.array([obs_x, obs_y])) < 
                obs_radius + self.robot_radius + c.MIN_FROM_ROBOT_TO_OBS_DIST_GENERATION for obs_x, obs_y, obs_radius in self.obstacles)):
                is_valid = False

        is_valid = False
        attempts = 0
        while not is_valid:
            attempts += 1

            if attempts > c.MAX_ATTEMPTS_FOR_WHILE_GENERATION:
                raise RuntimeError(f"Too many attempts to generate target")

            target_x = self.np_random.uniform(self.target_radius + c.MAP_EDGES_BUFFER_DURING_TARGET_GENERATION, 1 - self.target_radius)
            target_y = self.np_random.uniform(self.target_radius + c.MAP_EDGES_BUFFER_DURING_TARGET_GENERATION, 1 - self.target_radius)
            
            is_valid = True

            if np.linalg.norm(np.array([target_x, target_y]) - np.array([self.robot_x, self.robot_y])) < self.robot_radius + self.target_radius + c.MIN_FROM_ROBOT_TO_TARGET_DIST_GENERATION:
                is_valid = False

            if any((np.linalg.norm(np.array([target_x, target_y]) - np.array([obs_x, obs_y])) < obs_radius + self.target_radius + c.MIN_FROM_OBS_TO_TARGET_DIST_GENERATION for obs_x, obs_y, obs_radius in self.obstacles)):
                is_valid = False

        self.target_x = target_x
        self.target_y = target_y

        self.robot_speed = 0.0
        self.current_steering = 0.0
        self.robot_speed_x = 0
        self.robot_speed_y = 0
        self.robot_angle = self.np_random.uniform(-np.pi, np.pi)
        self.current_step = 0
        self.previous_action = c.ACTION_STAND

        current_distance = self._distance_to_target()
        current_ray_distances = self._get_ray_distances()
        current_min_ray_distance = current_ray_distances.min()

        self.prev_distance_to_target = current_distance
        self.prev_min_ray_distance = current_min_ray_distance
        self.delta_distance_to_target = 0.0
        self.delta_min_ray_distance = 0.0

        info = {
            "distance_to_target": current_distance,
            "is_success": False,
            "crashed": False,
            "steps_taken": self.current_step,
        }

        return self._get_obs(), info

    def step(self, action):
        old_distance = self._distance_to_target()
        old_ray_distances = self._get_ray_distances()
        old_min_ray_distance = old_ray_distances.min()

        action_config = c.DISCRETE_ACTIONS[int(action)]
        throttle = action_config["throttle"]
        steering = action_config["steering"]
        brake = action_config["brake"]

        self.current_steering = (
            (1.0 - c.ENV_STEERING_SMOOTHING_FACTOR) * self.current_steering
            + c.ENV_STEERING_SMOOTHING_FACTOR * steering
        )
        turn_amount = self.current_steering * c.ENV_MAX_TURN_RATE
        self.robot_angle += turn_amount
        self.robot_angle = (self.robot_angle + np.pi) % (2 * np.pi) - np.pi

        # Разгон
        self.robot_speed += throttle * c.ENV_ACCELERATION
        # Инерционное трение
        self.robot_speed *= 1.0 - c.ENV_FRICTION
        # Тормоз
        self.robot_speed *= 1.0 - brake * c.ENV_BRAKE_STRENGTH
        # Ограничение скорости
        self.robot_speed = np.clip(self.robot_speed, 0, c.ENV_MAX_FORWARD_SPEED)
        # Перевод скорости в x/y
        self.robot_speed_x = self.robot_speed * np.cos(self.robot_angle)
        self.robot_speed_y = self.robot_speed * np.sin(self.robot_angle)

        # Перемещение
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
        new_distance = self._distance_to_target()
        new_ray_distances = self._get_ray_distances()
        new_min_ray_distance = new_ray_distances.min()
        reached_target = new_distance < self.robot_radius + self.target_radius + c.FROM_ROBOT_TO_TARGET_DIST_ACCURACY_REGISTRATION

        self.delta_distance_to_target = old_distance - new_distance
        self.delta_min_ray_distance = old_min_ray_distance - new_min_ray_distance

        reward = 0.0
        distance_diff = self.delta_distance_to_target

        safe_distance = c.MIN_RAY_DISTANCE_TO_SAFE_ZONE_REGISTRATION
        in_safe_zone = new_min_ray_distance < safe_distance

        target_visible = self._is_target_visible()
        progress_coeff = c.DENSE_REWARD_COEFF if target_visible else c.DENSE_REWARD_COEFF * 0.2
        reward += distance_diff * progress_coeff

        reward -= c.TIME_PENALTY

        if in_safe_zone:
            proximity_factor = (safe_distance - new_min_ray_distance) / safe_distance
            reward -= proximity_factor * c.PROXIMITY_PENALTY_COEFF

            obstacle_approach = old_min_ray_distance - new_min_ray_distance
            if obstacle_approach > 0:
                reward -= obstacle_approach * c.OBSTACLE_APPROACH_PENALTY_COEFF

        if self.robot_speed < 0.0:
            reward -= c.BACKWARD_SPEED_PENALTY * abs(self.robot_speed)

        if abs(self.robot_speed) < c.MIN_MOVING_SPEED and not reached_target:
            reward -= c.STAND_STILL_PENALTY

        self.current_step += 1
        terminated = reached_target or crashed
        truncated = self.current_step >= self.max_steps

        if reached_target:
            reward += c.LARGE_REWARD
        elif crashed:
            reward -= c.LARGE_PENALTY
        elif truncated:
            reward -= c.TIMEOUT_PENALTY

        self.previous_action = int(action)
        next_obs = self._get_obs()
        self.prev_distance_to_target = new_distance
        self.prev_min_ray_distance = new_min_ray_distance

        info = {
            "distance_to_target": new_distance,
            "is_success": reached_target,
            "crashed": crashed,
            "steps_taken": self.current_step
        }

        return next_obs, reward, terminated, truncated, info

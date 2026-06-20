from robot_env import RobotEnv
import numpy as np

env = RobotEnv()
env.reset()
print(f"Количество препятствий: {len(env.obstacles)}")

flag = False
for i, (x1, y1, r1) in enumerate(env.obstacles):
    for j, (x2, y2, r2) in enumerate(env.obstacles):
        if i < j:  # Проверяем только уникальные пары
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            min_distance = r1 + r2 + env.MIN_FROM_OBS_TO_OBS_DIST  # Буфер
            if distance < min_distance:
                print(f"Препятствия {i} и {j} пересекаются!")
                flag = True

if not flag:
    print("OK!")
from robot_env import RobotEnv

env = RobotEnv()
obs, info = env.reset()
print(f"Начальная позиция: x={env.robot_x:.3f}, y={env.robot_y:.3f}")

for i in range(10):
    obs, reward, terminated, truncated, info = env.step(1)  # Вперёд
    print(f"Шаг {i+1}: x={env.robot_x:.3f}, y={env.robot_y:.3f}, reward={reward:.2f}")
    if terminated:
        print("Эпизод завершён!")
        break
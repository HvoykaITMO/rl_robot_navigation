from robot_env import RobotEnv

env = RobotEnv()

observation, info = env.reset()
print(f"Info: {info}")
print(f"Shape: {observation.shape}")
print(f"Min: {observation.min()}, Max: {observation.max()}")
print("Observation:", observation)
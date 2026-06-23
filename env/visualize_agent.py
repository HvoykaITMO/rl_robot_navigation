import pygame
import numpy as np
from pathlib import Path


from env.robot_env import RobotEnv
from agents.dqn_agent import DQNAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_dqn_model.pth"
WINDOW_SIZE = 1000
FPS = 30
ACTIONS = {
    0: 'Stand',
    1: 'Forward',
    2: 'Backward',
    3: 'Left',
    4: 'Right'
}
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)


def env_to_pygame(env_x, env_y):
    pygame_x = int(env_x * WINDOW_SIZE)
    pygame_y = int((1.0 - env_y) * WINDOW_SIZE)  # Инверсия Y
    return pygame_x, pygame_y


def env_radius_to_pixels(env_radius):
    return int(env_radius * WINDOW_SIZE)


def draw_text(screen, font, text, position, color=BLACK):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)


def render_env(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env: RobotEnv,
    observation: np.ndarray,
    reward: float,
    episode: int,
    episode_reward: float,
    action: int,
    info: dict,
):    
    # Отрисовка
        # Фон
    screen.fill(WHITE)
        # Границы карты
    pygame.draw.rect(screen, BLACK, (0, 0, WINDOW_SIZE, WINDOW_SIZE), width=3)

        # Препятствия
    for obs_x, obs_y, obs_radius in env.obstacles:
        center = env_to_pygame(obs_x, obs_y)
        radius = env_radius_to_pixels(obs_radius)
        pygame.draw.circle(screen, GRAY, center, radius)
        
        # Цель
    target_center = env_to_pygame(env.target_x, env.target_y)
    target_radius = env_radius_to_pixels(env.target_radius)
    pygame.draw.circle(screen, GREEN, target_center, target_radius)

        # Лучи
    for i, (end_x, end_y) in enumerate(env.ray_endpoints):
        start_point = env_to_pygame(env.robot_x, env.robot_y)
        end_point = env_to_pygame(end_x, end_y)
        ray_value = observation[7 + i]
        color = (
            int(255 * (1 - ray_value)),  # R: больше, если близко
            int(255 * ray_value),         # G: больше, если далеко
            0
        )
        pygame.draw.line(screen, color, start_point, end_point, width=2)

        # Робот
    robot_center = env_to_pygame(env.robot_x, env.robot_y)
    robot_radius = env_radius_to_pixels(env.robot_radius)
    pygame.draw.circle(screen, BLUE, robot_center, robot_radius)

        # Направление робота
    line_length = 0.03 # в нормализованных единицах
    end_x = env.robot_x + line_length * np.cos(env.robot_angle)
    end_y = env.robot_y + line_length * np.sin(env.robot_angle)
    start_point = env_to_pygame(env.robot_x, env.robot_y)
    end_point = env_to_pygame(end_x, end_y)
    pygame.draw.line(screen, RED, start_point, end_point, width=5)

    # Информация
    if info.get("is_success", False):
        status = "SUCCESS!"
        status_color = GREEN
    elif info.get("crashed", False):
        status = "CRASHED!"
        status_color = RED
    else:
        status = "Running"
        status_color = BLACK
    action_name = ACTIONS.get(action, "Unknown")
    draw_text(screen, font, f"Episode: {episode}", (10, 10))
    draw_text(screen, font, f"Step: {env.current_step}", (10, 40))
    draw_text(screen, font, f"Reward: {reward:.2f}", (10, 70))
    draw_text(screen, font, f"Episode reward: {episode_reward:.2f}", (10, 100))
    draw_text(screen, font, f"Action: {action_name}", (10, 130))
    draw_text(screen, font, f"Distance: {info['distance_to_target']:.3f}", (10, 160))
    draw_text(screen, font, f"Status: {status}", (10, 190), status_color)
    draw_text(screen, font, f"Robot: ({env.robot_x:.2f}, {env.robot_y:.2f})", (10, 220))

    pygame.display.flip()


def main():
    # Инициализация среды
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("DQN Agent Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24) # None = стандартный шрифт, 24 = размер

    # Создание среды
    env = RobotEnv(max_steps=500)
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_size=128
    )
    agent.load_model(str(MODEL_PATH))
    agent.epsilon = 0.0
    agent.q_net.eval()

    reward = 0.0
    episode = 0
    running = True

    while running:
        episode += 1
        state, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0

        while running and not terminated and not truncated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        terminated = True

            if not running or terminated:
                break

            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state

            render_env(screen, font, env, state, reward, episode, episode_reward, action, info)
            clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

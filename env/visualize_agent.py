import pygame
import numpy as np
import argparse
from pathlib import Path


from env.robot_env import RobotEnv
from agents.dqn_agent import DQNAgent
from utils import constants as c


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / c.MODEL_DIR / c.VISUALIZE_MODEL_FILENAME


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trained DQN agent")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to the .pth model file to visualize",
    )
    return parser.parse_args()


def env_to_pygame(env_x, env_y):
    pygame_x = int(env_x * c.WINDOW_SIZE)
    pygame_y = int((1 - env_y) * c.WINDOW_SIZE)  # Инверсия Y
    return pygame_x, pygame_y


def env_radius_to_pixels(env_radius):
    return int(env_radius * c.WINDOW_SIZE)


def draw_text(screen, font, text, position, color=c.BLACK):
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
    screen.fill(c.WHITE)
        # Границы карты
    pygame.draw.rect(screen, c.BLACK, c.MAP_RECT, width=c.MAP_BORDER_WIDTH)

        # Препятствия
    for obs_x, obs_y, obs_radius in env.obstacles:
        center = env_to_pygame(obs_x, obs_y)
        radius = env_radius_to_pixels(obs_radius)
        pygame.draw.circle(screen, c.GRAY, center, radius)
        
        # Цель
    target_center = env_to_pygame(env.target_x, env.target_y)
    target_radius = env_radius_to_pixels(env.target_radius)
    pygame.draw.circle(screen, c.GREEN, target_center, target_radius)

        # Лучи
    for i, (end_x, end_y) in enumerate(env.ray_endpoints):
        start_point = env_to_pygame(env.robot_x, env.robot_y)
        end_point = env_to_pygame(end_x, end_y)
        ray_value = observation[len(observation) - c.RAYS_AMOUNT_GENERATION + i]
        color = (
            int(255 * (1 - ray_value)),  # R: больше, если близко
            int(255 * ray_value),         # G: больше, если далеко
            0
        )
        pygame.draw.line(screen, color, start_point, end_point, width=c.RAY_LINE_WIDTH)

        # Робот
    robot_center = env_to_pygame(env.robot_x, env.robot_y)
    robot_radius = env_radius_to_pixels(env.robot_radius)
    pygame.draw.circle(screen, c.BLUE, robot_center, robot_radius)

        # Направление робота
    line_length = c.ROBOT_DIRECTION_LINE_LENGTH # в нормализованных единицах
    end_x = env.robot_x + line_length * np.cos(env.robot_angle)
    end_y = env.robot_y + line_length * np.sin(env.robot_angle)
    start_point = env_to_pygame(env.robot_x, env.robot_y)
    end_point = env_to_pygame(end_x, end_y)
    pygame.draw.line(screen, c.RED, start_point, end_point, width=c.ROBOT_DIRECTION_LINE_WIDTH)

    # Информация
    if info.get("is_success", False):
        status = c.STATUS_SUCCESS
        status_color = c.GREEN
    elif info.get("crashed", False):
        status = c.STATUS_CRASHED
        status_color = c.RED
    else:
        status = c.STATUS_RUNNING
        status_color = c.BLACK
    action_name = c.ACTION_NAMES.get(action, c.ACTION_UNKNOWN_NAME)
    draw_text(screen, font, f"Episode: {episode}", c.VISUALIZE_AGENT_EPISODE_TEXT_POS)
    draw_text(screen, font, f"Step: {env.current_step}", c.VISUALIZE_AGENT_STEP_TEXT_POS)
    draw_text(screen, font, f"Reward: {reward:.2f}", c.VISUALIZE_AGENT_REWARD_TEXT_POS)
    draw_text(screen, font, f"Episode reward: {episode_reward:.2f}", c.VISUALIZE_AGENT_EPISODE_REWARD_TEXT_POS)
    draw_text(screen, font, f"Action: {action_name}", c.VISUALIZE_AGENT_ACTION_TEXT_POS)
    draw_text(screen, font, f"Distance: {info['distance_to_target']:.3f}", c.VISUALIZE_AGENT_DISTANCE_TEXT_POS)
    draw_text(screen, font, f"Status: {status}", c.VISUALIZE_AGENT_STATUS_TEXT_POS, status_color)
    draw_text(screen, font, f"Robot: ({env.robot_x:.2f}, {env.robot_y:.2f})", c.VISUALIZE_AGENT_ROBOT_TEXT_POS)

    pygame.display.flip()


def main(model_path: Path):
    # Инициализация среды
    pygame.init()
    screen = pygame.display.set_mode((c.WINDOW_SIZE, c.WINDOW_SIZE))
    pygame.display.set_caption(c.DQN_AGENT_VISUALIZATION_CAPTION)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, c.PYGAME_FONT_SIZE) # None = стандартный шрифт

    # Создание среды
    env = RobotEnv(
        step_size=c.ENV_STEP_SIZE,
        turn_angle=c.ENV_TURN_ANGLE,
        max_steps=c.EPISODE_MAX_STEPS,
        num_obstacles=c.ENV_NUM_OBSTACLES,
        robot_radius=c.ENV_ROBOT_RADIUS,
        target_radius=c.ENV_TARGET_RADIUS,
    )
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_size=c.TRAIN_AGENT_HIDDEN_SIZE,
        learning_rate=c.TRAIN_LEARNING_RATE,
        gamma=c.TRAIN_GAMMA,
        epsilon=c.VISUALIZE_AGENT_EPSILON,
        epsilon_min=c.TRAIN_EPSILON_MIN,
        epsilon_decay=c.TRAIN_EPSILON_DECAY,
        buffer_size=c.TRAIN_BUFFER_SIZE,
        batch_size=c.TRAIN_BATCH_SIZE,
        target_update=c.TRAIN_TARGET_UPDATE,
        max_grad_norm=c.TRAIN_MAX_GRAD_NORM,
    )
    agent.load_model(str(model_path))
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
            clock.tick(c.FPS)

    pygame.quit()


if __name__ == "__main__":
    args = parse_args()
    main(args.model_path)

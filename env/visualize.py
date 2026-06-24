import pygame
import numpy as np
from env.robot_env import RobotEnv
from utils import constants as c


def env_to_pygame(env_x, env_y):
    pygame_x = int(env_x * c.WINDOW_SIZE)
    pygame_y = int((c.PYGAME_COORD_MAX - env_y) * c.WINDOW_SIZE)  # Инверсия Y
    return pygame_x, pygame_y


def env_radius_to_pixels(env_radius):
    return int(env_radius * c.WINDOW_SIZE)


def draw_text(screen, font, text, position, color=c.BLACK):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)


def main():
    # Инициализация среды
    pygame.init()
    screen = pygame.display.set_mode((c.WINDOW_SIZE, c.WINDOW_SIZE))
    pygame.display.set_caption(c.ROBOT_NAVIGATION_CAPTION)
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
    observation, info = env.reset()

    # Главный цикл
    running = True
    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Обработка клавиш
        keys = pygame.key.get_pressed() # Возвращает состояние клавиш в данный момент
        action = c.ACTION_STAND # По умолчанию стоять
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action = c.ACTION_FORWARD  # Вперёд
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action = c.ACTION_BACKWARD  # Назад
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = c.ACTION_LEFT  # Поворот влево
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = c.ACTION_RIGHT  # Поворот вправо
        
        # Применение действия
        if keys[pygame.K_r]: # Перезапуск по клавише
            obs, info = env.reset()
        else:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated: # Конец эпизода - перезапуск
                observation, info = env.reset()
        
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
            ray_value = observation[7 + i]
            color = (
                int(c.RAY_COLOR_MAX * (c.PYGAME_COORD_MAX - ray_value)),  # R: больше, если близко
                int(c.RAY_COLOR_MAX * ray_value),         # G: больше, если далеко
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
        end_y = env.robot_y + line_length * np.sin(env.robot_angle) # Минус для инверсии угла (нужна из-за env_to_pygame)
        start_point = env_to_pygame(env.robot_x, env.robot_y)
        end_point = env_to_pygame(end_x, end_y)
        pygame.draw.line(screen, c.RED, start_point, end_point, width=c.ROBOT_DIRECTION_LINE_WIDTH)

            # Информация
                # Определяем статус
        if info.get("is_success", False):
            status = c.STATUS_SUCCESS
            status_color = c.GREEN
        elif info.get("crashed", False):
            status = c.STATUS_CRASHED
            status_color = c.RED
        else:
            status = c.STATUS_RUNNING
            status_color = c.BLACK
                # Отрисовка текста
        draw_text(screen, font, f"Reward: {reward:.2f}", c.VISUALIZE_REWARD_TEXT_POS)
        draw_text(screen, font, f"Steps: {env.current_step}", c.VISUALIZE_STEPS_TEXT_POS)
        draw_text(screen, font, f"Distance: {info['distance_to_target']:.3f}", c.VISUALIZE_DISTANCE_TEXT_POS)
        draw_text(screen, font, f"Status: {status}", c.VISUALIZE_STATUS_TEXT_POS, status_color)
        draw_text(screen, font, f"Robot: ({env.robot_x:.2f}, {env.robot_y:.2f})", c.VISUALIZE_ROBOT_TEXT_POS)

        # Обновление экрана
        pygame.display.flip()
        clock.tick(c.FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

import pygame
import numpy as np
from robot_env import RobotEnv


WINDOW_SIZE = 1000
FPS = 30


def env_to_pygame(env_x, env_y):
    pygame_x = int(env_x * WINDOW_SIZE)
    pygame_y = int((1.0 - env_y) * WINDOW_SIZE)  # Инверсия Y
    return pygame_x, pygame_y


def env_radius_to_pixels(env_radius):
    return int(env_radius * WINDOW_SIZE)


def draw_text(screen, font, text, position, color=(0, 0, 0)):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)


def main():
    # Инициализация среды
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Robot Navigation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24) # None = стандартный шрифт, 24 = размер

    # Создание среды
    env = RobotEnv()
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
        action = 0 # По умолчанию стоять
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action = 1  # Вперёд
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action = 2  # Назад
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = 3  # Поворот влево
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = 4  # Поворот вправо
        
        # Применение действия
        if keys[pygame.K_r]: # Перезапуск по клавише
            obs, info = env.reset()
        else:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated: # Конец эпизода - перезапуск
                observation, info = env.reset()
        
        # Отрисовка
            # Фон
        screen.fill((255, 255, 255))

            # Границы карты
        pygame.draw.rect(screen, (0, 0, 0), (0, 0, WINDOW_SIZE, WINDOW_SIZE), width=3)

            # Препятствия
        for obs_x, obs_y, obs_radius in env.obstacles:
            center = env_to_pygame(obs_x, obs_y)
            radius = env_radius_to_pixels(obs_radius)
            pygame.draw.circle(screen, (128, 128, 128), center, radius)
            
            # Цель
        target_center = env_to_pygame(env.target_x, env.target_y)
        target_radius = env_radius_to_pixels(env.target_radius)
        pygame.draw.circle(screen, (0, 255, 0), target_center, target_radius)

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
        pygame.draw.circle(screen, (0, 0, 255), robot_center, robot_radius)

            # Направление робота
        line_length = 0.03 # в нормализованных единицах
        end_x = env.robot_x + line_length * np.cos(env.robot_angle)
        end_y = env.robot_y + line_length * np.sin(env.robot_angle) # Минус для инверсии угла (нужна из-за env_to_pygame)
        start_point = env_to_pygame(env.robot_x, env.robot_y)
        end_point = env_to_pygame(end_x, end_y)
        pygame.draw.line(screen, (255, 0, 0), start_point, end_point, width=5)

            # Информация

                # Определяем статус
        if info.get("is_success", False):
            status = "SUCCESS!"
            status_color = (0, 200, 0)
        elif info.get("crashed", False):
            status = "CRASHED!"
            status_color = (200, 0, 0)
        else:
            status = "Running"
            status_color = (0, 0, 0)
                # Отрисовка текста
        draw_text(screen, font, f"Reward: {reward:.2f}", (10, 10))
        draw_text(screen, font, f"Steps: {env.current_step}", (10, 40))
        draw_text(screen, font, f"Distance: {info['distance_to_target']:.3f}", (10, 70))
        draw_text(screen, font, f"Status: {status}", (10, 100), status_color)
        draw_text(screen, font, f"Robot: ({env.robot_x:.2f}, {env.robot_y:.2f})", (10, 130))

        # Обновление экрана
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

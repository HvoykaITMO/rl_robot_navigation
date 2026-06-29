# RL Robot Navigation

Проект с обучением DQN-агента для навигации робота в 2D-среде с препятствиями. Робот получает наблюдения из состояния среды и лучей до препятствий, выбирает дискретные действия управления и учится добираться до цели без столкновений.

## Демонстрация

<video src="RL_demonstration.mp4" controls width="700">
  Your browser does not support the video tag.
</video>

Если видео не отображается в просмотрщике Markdown, откройте файл [`RL_demonstration.mp4`](RL_demonstration.mp4) напрямую.

## Установка

Рекомендуется запускать из корня проекта:

```bash
pip install -r requirements.txt
```

## Команды запуска

Ручная визуализация среды:

```bash
python -m env.visualize
```

Управление: `W`/`Up` - газ, `A`/`D` или `Left`/`Right` - поворот, `Space` - тормоз, `R` - перезапуск эпизода.

Обучение DQN-агента:

```bash
python -m agents.train_dqn
```

Продолжить обучение из сохранённой модели:

```bash
python -m agents.train_dqn --load-model models/4_obstacles/last_v5.1.pth
```

Визуализация обученного агента:

```bash
python -m env.visualize_agent
```

Запуск визуализации с конкретной моделью:

```bash
python -m env.visualize_agent --model-path models/4_obstacles/last_v5.1.pth
```

## Настройки

Основные константы обучения, среды и визуализации находятся в [`utils/constants.py`](utils/constants.py).

Там можно менять:

- параметры среды: количество препятствий, радиусы робота/цели, лимит шагов эпизода;
- reward shaping: награды, штрафы, коэффициенты прогресса;
- гиперпараметры DQN: число эпизодов, learning rate, gamma, batch size, epsilon decay;
- параметры графиков и сохранения моделей;
- настройки окна Pygame и FPS визуализации.

После обучения модели сохраняются в папку `models`, а график результатов сохраняется как `Training_results.png`.
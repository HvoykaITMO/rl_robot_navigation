import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from env.robot_env import RobotEnv
from agents.dqn_agent import DQNAgent
from utils import constants as c


def train(
    num_episodes=c.TRAIN_NUM_EPISODES,
    max_steps=c.EPISODE_MAX_STEPS,
    load_path=None,
):
    env = RobotEnv(
        step_size=c.ENV_STEP_SIZE,
        turn_angle=c.ENV_TURN_ANGLE,
        max_steps=max_steps,
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
        buffer_size=c.TRAIN_BUFFER_SIZE,
        batch_size=c.TRAIN_BATCH_SIZE,
        target_update=c.TRAIN_TARGET_UPDATE,
        max_grad_norm=c.TRAIN_MAX_GRAD_NORM,
        epsilon=c.TRAIN_EPSILON,
        epsilon_min=c.TRAIN_EPSILON_MIN,
        epsilon_decay=c.TRAIN_EPSILON_DECAY
    )

    if load_path:
        agent.load_model(load_path)

    reward_history = []
    loss_history = []
    epsilon_history = []
    episode_length_history = []

    best_reward = -float("inf")
    os.makedirs(c.MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(c.MODEL_DIR, c.DQN_MODEL_FILENAME)

    pbar = tqdm(range(num_episodes), desc=c.TRAINING_PROGRESS_DESC)

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = []
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.remember(state, action, reward, next_state, terminated or truncated)
            loss = agent.learn()

            if loss > 0:
                episode_loss.append(loss)

            state = next_state
            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                break

        reward_history.append(episode_reward)
        episode_length_history.append(episode_length)
        loss_history.append(np.mean(episode_loss) if episode_loss else 0)
        epsilon_history.append(agent.epsilon)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(best_model_path)

        agent.decay_epsilon()

        pbar.set_postfix({
            "Reward": f"{episode_reward:.2f}",
            "Loss": f"{np.mean(episode_loss) if episode_loss else 0:.4f}",
            "Epsilon": f"{agent.epsilon:.4f}",
            "Length": f"{episode_length}",
            "Best": f"{best_reward:.2f}",
        })
        pbar.update(1)
    pbar.close()

    return reward_history, loss_history, epsilon_history, episode_length_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN agent for robot navigation")
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to an existing .pth model to continue training from",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reward_history, loss_history, epsilon_history, episode_length_history = train(
        load_path=args.load_model,
    )
    plt.figure(figsize=c.TRAINING_PLOT_FIGSIZE)
    plt.subplot(*c.TRAINING_PLOT_REWARD_SUBPLOT)
    plt.plot(reward_history, color=c.TRAINING_PLOT_REWARD_COLOR)
    plt.title(c.TRAINING_PLOT_REWARD_TITLE)
    plt.xlabel(c.TRAINING_PLOT_EPISODE_LABEL)
    plt.ylabel(c.TRAINING_PLOT_REWARD_LABEL)

    plt.subplot(*c.TRAINING_PLOT_LOSS_SUBPLOT)
    plt.plot(loss_history, color=c.TRAINING_PLOT_LOSS_COLOR)
    plt.title(c.TRAINING_PLOT_LOSS_TITLE)
    plt.xlabel(c.TRAINING_PLOT_EPISODE_LABEL)
    plt.ylabel(c.TRAINING_PLOT_LOSS_LABEL)

    plt.subplot(*c.TRAINING_PLOT_EPSILON_SUBPLOT)
    plt.plot(epsilon_history, color=c.TRAINING_PLOT_EPSILON_COLOR)
    plt.title(c.TRAINING_PLOT_EPSILON_TITLE)
    plt.xlabel(c.TRAINING_PLOT_EPISODE_LABEL)
    plt.ylabel(c.TRAINING_PLOT_EPSILON_LABEL)

    plt.subplot(*c.TRAINING_PLOT_LENGTH_SUBPLOT)
    plt.plot(episode_length_history, color=c.TRAINING_PLOT_LENGTH_COLOR)
    plt.title(c.TRAINING_PLOT_LENGTH_TITLE)
    plt.xlabel(c.TRAINING_PLOT_EPISODE_LABEL)
    plt.ylabel(c.TRAINING_PLOT_LENGTH_LABEL)

    plt.tight_layout()
    plt.savefig(os.path.join(c.MODEL_DIR, c.TRAINING_RESULTS_FILENAME))
    plt.show()

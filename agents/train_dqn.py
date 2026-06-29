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
    episode_length_history = []
    success_history = []
    success_rate_history = []

    os.makedirs(c.MODEL_DIR, exist_ok=True)

    best_model_path = os.path.join(c.MODEL_DIR, c.DQN_MODEL_FILENAME)
    last_model_path = os.path.join(c.MODEL_DIR, c.DQN_LAST_MODEL_FILENAME)

    pbar = tqdm(range(num_episodes), desc=c.TRAINING_PROGRESS_DESC)

    best_success_rate = -float("inf")
    best_mean_reward = -float("inf")

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
        success_history.append(1 if info["is_success"] else 0)

        reward_window = reward_history[-c.TRAIN_SUCCESS_RATE_WINDOW:]
        success_window = success_history[-c.TRAIN_SUCCESS_RATE_WINDOW:]
        success_rate = np.mean(success_window)
        success_rate_history.append(success_rate)
        mean_reward = np.mean(reward_window)
        if len(success_window) == c.TRAIN_SUCCESS_RATE_WINDOW:
            is_better_success = success_rate > best_success_rate
            is_same_success_better_reward = (
                success_rate == best_success_rate
                and mean_reward > best_mean_reward
            )
            if is_better_success or is_same_success_better_reward:
                best_success_rate = success_rate
                best_mean_reward = mean_reward
                agent.save_model(best_model_path)

        agent.decay_epsilon()

        pbar.set_postfix({
            "Reward": f"{episode_reward:.2f}",
            "MeanReward": f"{mean_reward:.2f}",
            "BestReward": f"{max(best_mean_reward, 0):.2f}",
            "Loss": f"{np.mean(episode_loss) if episode_loss else 0:.4f}",
            "Epsilon": f"{agent.epsilon:.4f}",
            "Length": f"{episode_length}",
            "SuccessRate": f"{success_rate:.2%}",
            "BestSuccess": f"{max(best_success_rate, 0):.2%}",
        })
        pbar.update(1)
    pbar.close()
    agent.save_model(last_model_path)

    return reward_history, loss_history, success_rate_history, episode_length_history


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
    reward_history, loss_history, success_rate_history, episode_length_history = train(
        load_path=args.load_model,
    )
    plt.figure(figsize=c.TRAINING_PLOT_FIGSIZE)
    plt.subplot(*c.TRAINING_PLOT_REWARD_SUBPLOT)
    plt.scatter(range(len(reward_history)), reward_history, color=c.TRAINING_PLOT_REWARD_COLOR)
    plt.title(c.TRAINING_PLOT_REWARD_TITLE)
    plt.xlabel(c.TRAINING_PLOT_EPISODE_LABEL)
    plt.ylabel(c.TRAINING_PLOT_REWARD_LABEL)

    plt.subplot(*c.TRAINING_PLOT_LOSS_SUBPLOT)
    plt.plot(loss_history, color=c.TRAINING_PLOT_LOSS_COLOR)
    plt.title(c.TRAINING_PLOT_LOSS_TITLE)
    plt.xlabel(c.TRAINING_PLOT_EPISODE_LABEL)
    plt.ylabel(c.TRAINING_PLOT_LOSS_LABEL)

    plt.subplot(*c.TRAINING_PLOT_SUCCESS_RATE_SUBPLOT)
    plt.plot(success_rate_history, color=c.TRAINING_PLOT_SUCCESS_RATE_COLOR)
    plt.title(c.TRAINING_PLOT_SUCCESS_RATE_TITLE)
    plt.xlabel(c.TRAINING_PLOT_EPISODE_LABEL)
    plt.ylabel(c.TRAINING_PLOT_SUCCESS_RATE_LABEL)

    plt.subplot(*c.TRAINING_PLOT_LENGTH_SUBPLOT)
    plt.plot(episode_length_history, color=c.TRAINING_PLOT_LENGTH_COLOR)
    plt.title(c.TRAINING_PLOT_LENGTH_TITLE)
    plt.xlabel(c.TRAINING_PLOT_EPISODE_LABEL)
    plt.ylabel(c.TRAINING_PLOT_LENGTH_LABEL)

    plt.tight_layout()
    plt.savefig(os.path.join(c.MODEL_DIR, c.TRAINING_RESULTS_FILENAME))
    plt.show()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from env.robot_env import RobotEnv
from agents.dqn_agent import DQNAgent


def train(num_episodes=5000, max_steps=500):
    env = RobotEnv()

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_size=128,
        learning_rate=0.0005,
        gamma=0.95,
        buffer_size=50000,
        batch_size=128,
        target_update=700,
        max_grad_norm=10.0,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995
    )

    reward_history = []
    loss_history = []
    epsilon_history = []
    episode_length_history = []

    best_reward = -float('inf')
    os.makedirs("models", exist_ok=True)

    pbar = tqdm(range(num_episodes), desc="Training")

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
            agent.save_model('models/best_dqn_model.pth')
        
        agent.decay_epsilon()

        pbar.set_postfix({
            'Reward': f'{episode_reward:.2f}',
            'Loss': f'{np.mean(episode_loss) if episode_loss else 0:.4f}',
            'Epsilon': f'{agent.epsilon:.4f}',
            'Length': f'{episode_length}',
            'Best Reward': f'{best_reward:.2f}'
        })
        pbar.update(1)
    pbar.close()

    return reward_history, loss_history, epsilon_history, episode_length_history


if __name__ == "__main__":
    reward_history, loss_history, epsilon_history, episode_length_history = train()
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(reward_history, color='blue')
    plt.title('Reward History')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(2, 2, 2)
    plt.plot(loss_history, color='orange')
    plt.title('Loss History')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 3)
    plt.plot(epsilon_history, color='red')
    plt.title('Epsilon History')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.subplot(2, 2, 4)
    plt.plot(episode_length_history, color='green')
    plt.title('Episode Length History')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')

    plt.tight_layout()
    plt.grid(True)
    plt.savefig('models/training_results.png')
    plt.show()

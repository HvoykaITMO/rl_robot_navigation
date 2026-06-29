import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from agents.q_network import QNetwork
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        learning_rate: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        buffer_size: int,
        batch_size: int,
        target_update: int,
        max_grad_norm: float,
    ):
        self.q_net = QNetwork(state_size, action_size, hidden_size)
        self.target_net = QNetwork(state_size, action_size, hidden_size)
        self.upgrade_target_network()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size)
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.max_grad_norm = max_grad_norm
        self.total_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def learn(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        q_predicted = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            # Double DQN
            best_next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            best_next_q_values = self.target_net(next_states).gather(1, best_next_actions)
            target_q = rewards + self.gamma * best_next_q_values * (1 - dones)
        loss = nn.SmoothL1Loss()(q_predicted, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.target_update == 0:
            self.upgrade_target_network()

        return loss.item()

    def upgrade_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load_model(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        self.upgrade_target_network()

    def save_model(self, path: str):
        torch.save(self.q_net.state_dict(), path)

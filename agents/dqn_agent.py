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
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 5000,
        batch_size: int = 32,
        target_update: int = 100,
        max_grad_norm: float = 10.0
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
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.max(dim=1, keepdim=True).values
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        loss = nn.MSELoss()(q_predicted, target_q)
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
        self.q_net.load_state_dict(torch.load(path))
        self.upgrade_target_network()

    def save_model(self, path: str):
        torch.save(self.q_net.state_dict(), path)


if __name__ == "__main__":
    agent = DQNAgent(
        state_size=15,
        action_size=5,
        hidden_size=64,
        buffer_size=1000,
        batch_size=32,
        target_update=10
    )
    
    print(f"Initial epsilon: {agent.epsilon}")
    

    for i in range(100):
        state = np.random.randn(15)
        action = random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.random.randn(15)
        done = random.random() < 0.1
        agent.remember(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(agent.buffer)}")
    

    test_state = np.random.randn(15)
    for i in range(5):
        action = agent.select_action(test_state)
        print(f"Action {i}: {action} (epsilon={agent.epsilon:.3f})")
    
    
    print("\nTraining...")
    for i in range(20):
        loss = agent.learn()
        if i % 5 == 0:
            print(f"Step {i}: loss={loss:.4f}, epsilon={agent.epsilon:.3f}")
    
    print(f"\nFinal epsilon: {agent.epsilon}")
    print("Test completed!")

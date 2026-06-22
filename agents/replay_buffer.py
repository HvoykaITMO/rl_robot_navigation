import torch
import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=100)
    for i in range(150):
        state = np.random.randn(15)
        action = random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.random.randn(15)
        done = random.random() < 0.1
        buffer.add(state, action, reward, next_state, done)
    batch = buffer.sample(32)
    states_t, actions_t, rewards_t, next_states_t, dones_t = batch
    print(states_t.shape)
    print(actions_t.shape)
    print(rewards_t.shape)
    print(next_states_t.shape)
    print(dones_t.shape)
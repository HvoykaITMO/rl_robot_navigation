import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_size=15, output_size=5, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    q_net = QNetwork()

    # Один observation без батча
    obs_single = torch.randn(15)
    print("Single:", q_net(obs_single).shape)  # должно быть (5,)

    # Один observation с батчем (как у тебя)
    obs_batch_1 = torch.randn(1, 15)
    print("Batch 1:", q_net(obs_batch_1).shape)  # должно быть (1, 5)

    # Батч из 32 observations
    obs_batch_32 = torch.randn(32, 15)
    print("Batch 32:", q_net(obs_batch_32).shape)  # должно быть (32, 5)
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """Simple MLP for MNIST: 784 -> 128 -> 10."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)

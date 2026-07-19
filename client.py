import copy

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Client:
    """Trains a local copy of the global model on this client's data shard."""

    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset

    def train(self, global_model, epochs, batch_size, learning_rate):
        model = copy.deepcopy(global_model)
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        model.train()
        for _ in range(epochs):
            for x, y in loader:
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        return model.state_dict(), len(self.dataset)

import torch
import torch.nn as nn
import torch.optim as optim
from privacy import clip_and_add_noise
from model import NeuralNetwork

class Client:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Added use_dp and noise_multiplier to the parameters here:
    def train(self, global_model, local_epochs, batch_size, lr, use_dp=True, noise_multiplier=0.05):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_model = NeuralNetwork().to(device)
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()

        initial_weights = {k: v.clone() for k, v in global_model.state_dict().items()}
        optimizer = optim.SGD(local_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(local_epochs):
            for x, y in self.dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = local_model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        local_weights = local_model.state_dict()

        # Apply Differential Privacy if enabled
        if use_dp:
            local_weights = clip_and_add_noise(
                local_weights, 
                initial_weights, 
                max_grad_norm=1.0, 
                noise_multiplier=noise_multiplier
            )

        dataset_size = len(self.dataloader.dataset)
        return local_weights, dataset_size
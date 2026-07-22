import torch
import torch.nn as nn
import torch.optim as optim
from model import NeuralNetwork
from dataset import load_mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the saved global model weights
net = NeuralNetwork().to(device)
net.load_state_dict(torch.load("FL--main/global_model.pth", map_location=device))
net.eval()

# 2. Simulate a client step to get target weight updates (Delta W)
criterion = nn.CrossEntropyLoss()
train_dataset, _ = load_mnist()
private_x, private_y = next(iter(torch.utils.data.DataLoader(
    torch.utils.data.Subset(train_dataset, [0]), batch_size=1
)))
private_x, private_y = private_x.to(device), private_y.to(device)

initial_weights = {k: v.clone() for k, v in net.state_dict().items()}

# Simulate local client update (1 step)
net.zero_grad()
output = net(private_x)
loss = criterion(output, private_y)
loss.backward()

optimizer_client = optim.SGD(net.parameters(), lr=0.01)
optimizer_client.step()

target_deltas = {k: net.state_dict()[k] - initial_weights[k] for k in initial_weights}

# 3. Attacker Optimization Loop
dummy_x = torch.randn(1, 1, 28, 28, requires_grad=True, device=device)
dummy_label_logits = torch.randn(1, 10, requires_grad=True, device=device)

optimizer_attacker = optim.Adam([dummy_x, dummy_label_logits], lr=0.1)

print("Starting gradient/parameter inversion attack on your model...")
for iteration in range(400):
    optimizer_attacker.zero_grad()
    
    # Re-instantiate model with initial weights
    attacker_model = NeuralNetwork().to(device)
    attacker_model.load_state_dict(initial_weights)
    
    dummy_pred = attacker_model(dummy_x)
    dummy_y = dummy_label_logits.softmax(dim=-1)
    
    dummy_loss = torch.sum(-dummy_y * torch.log_softmax(dummy_pred, dim=-1))
    
    # Use create_graph=True so gradients can flow back through the model parameters to dummy_x
    dummy_grads = torch.autograd.grad(dummy_loss, attacker_model.parameters(), create_graph=True)
    
    # Calculate dummy parameter updates dynamically while maintaining graph connectivity
    weight_diff = 0.0
    param_idx = 0
    for k in initial_weights:
        param = list(attacker_model.parameters())[param_idx]
        grad = dummy_grads[param_idx]
        param_idx += 1
        
        # Simulate SGD step functionally: W_new = W_old - lr * grad
        dummy_delta = (param - 0.01 * grad) - initial_weights[k]
        weight_diff = weight_diff + torch.sum((dummy_delta - target_deltas[k]) ** 2)
        
    weight_diff.backward()
    optimizer_attacker.step()
    
    if (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration + 1}, Loss: {weight_diff.item():.6f}")

print("\nAttack complete. dummy_x optimized.")
import torch

from client import Client
from dataset import load_mnist, split_clients
from model import NeuralNetwork
from server import Server
import os

ROUNDS = 1
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
NUM_CLIENTS = 3
USE_DP = True          # <--- Enable Differential Privacy here
NOISE_MULTIPLIER = 0.05 # <--- Control noise intensity

def save_checkpoint(model, filename="global_model.pth"):
    torch.save(model.state_dict(), filename)
    return filename

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def main():
    train_dataset, test_loader = load_mnist()
    client_data = split_clients(train_dataset, num_clients=NUM_CLIENTS)

    global_model = NeuralNetwork()

    # Load starting model weights if they exist, otherwise initialize fresh
    if os.path.exists("global_model.pth"):
        global_model.load_state_dict(torch.load("global_model.pth"))
        
    server = Server(global_model)
    clients = [Client(i, client_data[i]) for i in range(NUM_CLIENTS)]

    print(f"Clients: {NUM_CLIENTS} | Rounds: {ROUNDS} | Local epochs: {LOCAL_EPOCHS} | DP Enabled: {USE_DP}")
    for i, ds in enumerate(client_data):
        print(f"  Client {i}: {len(ds)} samples")

    for r in range(ROUNDS):
        print(f"\nRound {r + 1}/{ROUNDS}")

        client_models = []
        client_sizes = []
        for client in clients:
            # Passes the privacy parameters down into the client training routine
            weights, size = client.train(
                server.global_model, LOCAL_EPOCHS, BATCH_SIZE, LR, 
                use_dp=USE_DP, noise_multiplier=NOISE_MULTIPLIER
            )
            client_models.append(weights)
            client_sizes.append(size)

        server.aggregate(client_models, client_sizes)
        acc = evaluate(server.global_model, test_loader)
        print(f"  Test accuracy: {acc:.2f}%")

    saved_path = save_checkpoint(server.global_model, filename="global_model.pth")
    print(f"\nTraining complete! Global model saved to: {saved_path}")

if __name__ == "__main__":
    main()
import torch

from client import Client
from dataset import load_mnist, split_clients
from model import NeuralNetwork
from server import Server

ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
NUM_CLIENTS = 3


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
    server = Server(global_model)
    clients = [Client(i, client_data[i]) for i in range(NUM_CLIENTS)]

    print(f"Clients: {NUM_CLIENTS} | Rounds: {ROUNDS} | Local epochs: {LOCAL_EPOCHS}")
    for i, ds in enumerate(client_data):
        print(f"  Client {i}: {len(ds)} samples")

    for r in range(ROUNDS):
        print(f"\nRound {r + 1}/{ROUNDS}")

        client_models = []
        client_sizes = []
        for client in clients:
            weights, size = client.train(
                server.global_model, LOCAL_EPOCHS, BATCH_SIZE, LR
            )
            client_models.append(weights)
            client_sizes.append(size)

        server.aggregate(client_models, client_sizes)
        acc = evaluate(server.global_model, test_loader)
        print(f"  Test accuracy: {acc:.2f}%")

    print("\nFinished")


if __name__ == "__main__":
    main()

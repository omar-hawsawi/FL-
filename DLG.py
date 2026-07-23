
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import NeuralNetwork
from dataset import load_mnist

try:
    from torch.func import functional_call
except ImportError:  # older torch versions
    from torch.nn.utils.stateless import functional_call

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_PATH = "global_model.pth"          # path to the saved global model
LOCAL_LR = 0.01                          # lr the client used for its local step
NUM_STEPS = 1                            # keep at 1 for the classic DLG assumption

# Plain SGD (no momentum) on purpose, NOT Adam. Adam is adaptive -- it finds
# the descent direction and then converges in a fast, uneven burst (usually
# within a few hundred iterations for this tiny MLP), so every checkpoint from
# 500 onward ends up looking identical (already-solved). Plain SGD moves at a
# steady, predictable rate proportional to the raw gradient, which is what
# gives a smooth, visible progression spread across the full 10,000
# iterations -- matching the classic "Iters=0/10/50/100/500" figure from the
# DLG paper, just stretched to this script's checkpoint spacing.
ATTACK_LR = 20                           # attacker's SGD lr (tuned so the walk
                                          # from noise to the original takes
                                          # roughly the full 10k-iteration budget)
# TV (total-variation) regularization smooths the reconstructed image. It's
# useful for natural photos, but for tiny 28x28 MNIST digits it is actively
# harmful: the gradient-matching loss here operates on a MUCH smaller scale
# (~1e-5) than a raw TV term (~hundreds), so even TV_WEIGHT=1e-3 completely
# dominates the optimization and the image gets pulled toward a smooth blob
# instead of matching the real gradients -- this was the exact cause of the
# loss plateauing at ~0.003 with no visible digit forming. Verified empirically:
#   TV_WEIGHT=1e-3  -> loss stuck ~0.0017,  pixel MSE ~0.09  (garbage)
#   TV_WEIGHT=0     -> loss ~0.00003,       pixel MSE ~0.00003 (near-perfect)
# Keep this at 0 (or at most ~1e-7) for MNIST-scale reconstructions.
TV_WEIGHT = 0.0
SEED = 0                                 # which private sample to attack (index in MNIST)

# Checkpoints at which we snapshot the reconstruction (grows 500 -> 1500 -> 3000 -> 5000)
CHECKPOINTS = [500, 2000, 5000, 10000]
TOTAL_ITERS = max(CHECKPOINTS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

# --------------------------------------------------------------------------
# 1. Load the real global model architecture + weights
# --------------------------------------------------------------------------
net = NeuralNetwork().to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.train()

# --------------------------------------------------------------------------
# 2. Pick a private sample and simulate the client's local update
#    (this produces the gradient/delta the attacker is assumed to see)
# --------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
train_dataset, _ = load_mnist()
private_x, private_y = next(iter(torch.utils.data.DataLoader(
    torch.utils.data.Subset(train_dataset, [SEED]), batch_size=1
)))
private_x, private_y = private_x.to(device), private_y.to(device)

initial_weights = {k: v.clone() for k, v in net.state_dict().items()}

optimizer_client = optim.SGD(net.parameters(), lr=LOCAL_LR)
for _ in range(NUM_STEPS):
    net.zero_grad()
    output = net(private_x)
    loss = criterion(output, private_y)
    loss.backward()
    optimizer_client.step()

target_deltas = {k: net.state_dict()[k] - initial_weights[k] for k in initial_weights}

# --------------------------------------------------------------------------
# 3. Infer the true label analytically from the last bias layer (iDLG trick)
#    Fixing the label (instead of optimizing it) gives much cleaner, more
#    stable convergence than treating it as unknown.
# --------------------------------------------------------------------------
bias_keys = [k for k in target_deltas.keys() if "bias" in k]
last_bias_key = bias_keys[-1]
true_label_idx = torch.argmax(target_deltas[last_bias_key]).item()
print(f"Inferred true label: {true_label_idx} (actual label was {private_y.item()})")

dummy_label = torch.zeros(1, 10, device=device)
dummy_label[0, true_label_idx] = 1.0  # fixed, not optimized

# --------------------------------------------------------------------------
# 4. Attacker optimization loop
# --------------------------------------------------------------------------
dummy_x = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)
optimizer_attacker = optim.SGD([dummy_x], lr=ATTACK_LR, momentum=0.0)

model_template = NeuralNetwork().to(device)  # structure only, params passed functionally
params = {k: v.detach().clone().requires_grad_(True) for k, v in initial_weights.items()}
param_items = list(params.items())

print(f"Starting DLG attack | device={device} | total iterations={TOTAL_ITERS}")
print(f"Checkpoints: {CHECKPOINTS}\n")

loss_history = []
checkpoint_results = []   # list of dicts: {"iter": int, "loss": float, "image": np.ndarray}
best_loss = float("inf")
best_image = None
checkpoint_set = set(CHECKPOINTS)

for iteration in range(1, TOTAL_ITERS + 1):
    optimizer_attacker.zero_grad()

    dummy_pred = functional_call(model_template, params, (dummy_x,))
    dummy_loss = criterion(dummy_pred, dummy_label.argmax(dim=1))

    dummy_grads = torch.autograd.grad(
        dummy_loss, [p for _, p in param_items], create_graph=True
    )

    weight_diff = 0.0
    for (k, _), grad in zip(param_items, dummy_grads):
        # functional single-step SGD update of the *fixed* initial weights
        dummy_delta = -LOCAL_LR * grad
        weight_diff = weight_diff + torch.sum((dummy_delta - target_deltas[k]) ** 2)

    tv_loss = (
        torch.sum(torch.abs(dummy_x[:, :, :, :-1] - dummy_x[:, :, :, 1:]))
        + torch.sum(torch.abs(dummy_x[:, :, :-1, :] - dummy_x[:, :, 1:, :]))
    )

    total_loss = weight_diff + TV_WEIGHT * tv_loss
    total_loss.backward()
    optimizer_attacker.step()

    dummy_x.data = torch.clamp(dummy_x.data, 0, 1)

    current_loss = weight_diff.item()
    loss_history.append(current_loss)

    if current_loss < best_loss:
        best_loss = current_loss
        best_image = dummy_x.detach().clone()

    if iteration in checkpoint_set:
        img = dummy_x.detach().clone().cpu().squeeze().numpy()
        checkpoint_results.append({"iter": iteration, "loss": current_loss, "image": img})
        print(f"[Checkpoint] Iteration {iteration:>6} | Loss: {current_loss:.8f}")

print(f"\nAttack complete. Best loss reached: {best_loss:.8f}")

# --------------------------------------------------------------------------
# 5. Visualization: original + 4 checkpoints (large, no loss-curve panel)
# --------------------------------------------------------------------------
n_panels = 1 + len(checkpoint_results)  # original + checkpoints
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6.5))

axes[0].imshow(private_x.cpu().squeeze().numpy(), cmap="gray")
axes[0].set_title(f"Original\nLabel: {private_y.item()}", fontsize=14)
axes[0].axis("off")

for i, res in enumerate(checkpoint_results, start=1):
    axes[i].imshow(res["image"], cmap="gray")
    axes[i].set_title(f"Iteration {res['iter']}\nLoss: {res['loss']:.6f}", fontsize=14)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("dlg_attack_result.png", dpi=150)
plt.show()

print("\nSaved figure to 'dlg_attack_result.png'")
print("\nSummary:")
for res in checkpoint_results:
    print(f"  - Iteration {res['iter']:>6}: Loss = {res['loss']:.8f}")
print(f"  - Best loss overall: {best_loss:.8f}")

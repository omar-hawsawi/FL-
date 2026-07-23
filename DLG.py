"""import torch
import torch.nn as nn
import torch.optim as optim
from model import NeuralNetwork
from dataset import load_mnist
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the saved global model weights
net = NeuralNetwork().to(device)
net.load_state_dict(torch.load("FL--the-code-that-sve-the-model/global_model.pth", map_location=device))
net.train()

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

bias_keys = [k for k in target_deltas.keys() if 'bias' in k]
last_bias_key = bias_keys[-1]
true_label_idx = torch.argmax(target_deltas[last_bias_key]).item()
print(f" Inferred true label: {true_label_idx}")

# 3. Attacker Optimization Loop
dummy_x = torch.randn(1, 1, 28, 28, requires_grad=True, device=device)
dummy_label = torch.zeros(10, device=device)
dummy_label[true_label_idx] = 1.0
dummy_label_logits = dummy_label.clone().requires_grad_(True)


optimizer_attacker = optim.Adam([dummy_x, dummy_label_logits], lr=0.5)

scheduler = optim.lr_scheduler.StepLR(optimizer_attacker, step_size=200, gamma=0.5)


print("Starting gradient/parameter inversion attack on your model...")


first_image = None      
first_loss = None       
final_image = None      
final_loss = None       
best_loss = float('inf')
best_image = None
loss_history = []  

for iteration in range(2000):
    optimizer_attacker.zero_grad()
    
    # Re-instantiate model with initial weights
    attacker_model = NeuralNetwork().to(device)
    attacker_model.load_state_dict(initial_weights)
    attacker_model.train()
    
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

    tv_loss = torch.sum(torch.abs(dummy_x[:, :, :, :-1] - dummy_x[:, :, :, 1:])) + \
              torch.sum(torch.abs(dummy_x[:, :, :-1, :] - dummy_x[:, :, 1:, :]))
    
    total_loss = weight_diff + 0.001 * tv_loss 

    total_loss.backward()
    optimizer_attacker.step()
    scheduler.step()

    dummy_x.data = torch.clamp(dummy_x.data, 0, 1)

    if iteration == 0:
        first_image = dummy_x.detach().clone().cpu().squeeze().numpy()
        first_loss = weight_diff.item()
        print(f"First iteration (0): Loss = {first_loss:.6f}")
    
    
    current_loss = weight_diff.item()
    loss_history.append(current_loss)
    
    
    if current_loss < best_loss:
        best_loss = current_loss
        best_image = dummy_x.detach().clone()

    if (iteration + 1) % 200 == 0:
        print(f"Iteration {iteration + 1}, Loss: {weight_diff.item():.6f}")

print("\nAttack complete. dummy_x optimized.")

final_image = dummy_x.detach().clone().cpu().squeeze().numpy()
final_loss = current_loss

print(f"\n Summary:")
print(f"  - First loss: {first_loss:.6f}")
print(f"  - Final loss: {final_loss:.6f}")
print(f"  - Best loss: {best_loss:.6f}")
print(f"  - Improvement: {(first_loss - final_loss):.6f}")

plt.figure(figsize=(20, 5))

plt.subplot(1, 5, 1)
plt.imshow(private_x.cpu().squeeze().numpy(), cmap='gray')
plt.title(f'Original\nLabel: {private_y.item()}')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(first_image, cmap='gray')
plt.title(f'First Iteration (0)\nLoss: {first_loss:.6f}')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(final_image, cmap='gray')
plt.title(f'Final Iteration (2000)\nLoss: {final_loss:.6f}')
plt.axis('off')

plt.subplot(1, 5, 4)
if best_image is not None:
    plt.imshow(best_image.cpu().squeeze().numpy(), cmap='gray')
    plt.title(f'Best Reconstruction\nLoss: {best_loss:.6f}')
else:
    plt.text(0.5, 0.5, 'No best image', ha='center', va='center')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.yscale('log')
plt.grid(True)
plt.axhline(y=best_loss, color='r', linestyle='--', label=f'Best: {best_loss:.6f}')
plt.legend()
plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='First')
plt.axvline(x=2000, color='orange', linestyle='--', alpha=0.5, label='Final')
plt.legend()

plt.tight_layout()
plt.savefig('dlg_comparison.png', dpi=150)
plt.show()

"""




"""
import torch
import torch.nn as nn
import torch.optim as optim
from model import NeuralNetwork
from dataset import load_mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the saved global model weights
net = NeuralNetwork().to(device)
net.load_state_dict(torch.load("FL--the-code-that-sve-the-model/global_model.pth", map_location=device))
net.train()  # ✅ التعديل 1: استخدم train() بدلاً من eval()

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

# ✅ التعديل 2: استنتاج التسمية الحقيقية من التدرجات (iDLG trick)
# ابحث عن أي وزن يحتوي على 'weight'
weight_keys = [k for k in target_deltas.keys() if 'weight' in k]
last_weight_key = weight_keys[-1]  # استخدم آخر طبقة
true_label_idx = torch.argmax(target_deltas[last_weight_key]).item()

print(f"✅ Inferred true label: {true_label_idx}")  # للتأكد

# 3. Attacker Optimization Loop
dummy_x = torch.randn(1, 1, 28, 28, requires_grad=True, device=device)

# ✅ التعديل 3: استخدم التسمية الحقيقية بدلاً من التخمين العشوائي
dummy_label = torch.zeros(10, device=device)
dummy_label[true_label_idx] = 1.0
dummy_label_logits = dummy_label.clone().requires_grad_(True)

# ✅ التعديل 4: تحسين معلمات الهجوم
optimizer_attacker = optim.Adam([dummy_x, dummy_label_logits], lr=0.5)  # زيادة من 0.1 إلى 0.5

# ✅ التعديل 5: إضافة جدول لتقليل معدل التعلم
scheduler = optim.lr_scheduler.StepLR(optimizer_attacker, step_size=200, gamma=0.5)

print("Starting gradient/parameter inversion attack on your model...")
print(f"Target label: {true_label_idx}")

best_loss = float('inf')
best_image = None

for iteration in range(2000):  # ✅ التعديل 6: زيادة التكرارات إلى 2000
    optimizer_attacker.zero_grad()
    
    # Re-instantiate model with initial weights
    attacker_model = NeuralNetwork().to(device)
    attacker_model.load_state_dict(initial_weights)
    attacker_model.train()  # ✅ التعديل 7: تأكد من وضع التدريب
    
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
    
    # ✅ التعديل 8: إضافة Total Variation regularization لتحسين جودة الصورة
    tv_loss = torch.sum(torch.abs(dummy_x[:, :, :, :-1] - dummy_x[:, :, :, 1:])) + \
              torch.sum(torch.abs(dummy_x[:, :, :-1, :] - dummy_x[:, :, 1:, :]))
    
    total_loss = weight_diff + 0.001 * tv_loss  # وزن صغير لتنعيم الصورة
    
    total_loss.backward()
    optimizer_attacker.step()
    scheduler.step()
    
    # ✅ التعديل 9: قيد القيم بين 0 و 1 (لأن الصور في MNIST بين 0 و 1)
    dummy_x.data = torch.clamp(dummy_x.data, 0, 1)
    
    # حفظ أفضل صورة
    current_loss = weight_diff.item()
    if current_loss < best_loss:
        best_loss = current_loss
        best_image = dummy_x.detach().clone()
    
    if (iteration + 1) % 200 == 0:
        print(f"Iteration {iteration + 1}, Loss: {current_loss:.8f}, TV: {tv_loss.item():.4f}")

print(f"\n✅ Attack completed! Best loss: {best_loss:.8f}")

# ✅ التعديل 10: حفظ وإظهار النتائج
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(private_x.cpu().squeeze().numpy(), cmap='gray')
plt.title(f'Original\nLabel: {private_y.item()}')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(best_image.cpu().squeeze().numpy(), cmap='gray')
plt.title(f'Best Reconstruction\nLoss: {best_loss:.6f}')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(dummy_x.detach().cpu().squeeze().numpy(), cmap='gray')
plt.title('Final Reconstruction')
plt.axis('off')

plt.tight_layout()
plt.savefig('dlg_improved_result.png', dpi=150)
plt.show()

print("\n📁 Saved result to 'dlg_improved_result.png'")

"""


"""
Deep Leakage from Gradients (DLG) attack against the saved federated-learning
global model.
 
Threat model (classic DLG / iDLG, Zhu et al. 2019):
    The attacker observes ONE client's weight update that resulted from a
    SINGLE local SGD step on a SINGLE private sample, and tries to
    reconstruct that sample purely from the gradient/weight-delta.
 
IMPORTANT NOTE ON REALISM:
    If a client trains for many local epochs/batches (like client.py does),
    the resulting delta is an aggregate of many different SGD steps and no
    longer matches this attack's linear "one-step" assumption -- reconstruction
    quality collapses in that regime (this is a known, expected limitation of
    DLG, not a bug). To get a clean, convincing demonstration, this script
    simulates the textbook single-step scenario explicitly (see NUM_STEPS
    below). Feel free to raise NUM_STEPS to see the degradation for yourself.
"""

"""
Deep Leakage from Gradients (DLG) attack against the saved federated-learning
global model.

Threat model (classic DLG / iDLG, Zhu et al. 2019):
    The attacker observes ONE client's weight update that resulted from a
    SINGLE local SGD step on a SINGLE private sample, and tries to
    reconstruct that sample purely from the gradient/weight-delta.

IMPORTANT NOTE ON REALISM:
    If a client trains for many local epochs/batches (like client.py does),
    the resulting delta is an aggregate of many different SGD steps and no
    longer matches this attack's linear "one-step" assumption -- reconstruction
    quality collapses in that regime (this is a known, expected limitation of
    DLG, not a bug). To get a clean, convincing demonstration, this script
    simulates the textbook single-step scenario explicitly (see NUM_STEPS
    below). Feel free to raise NUM_STEPS to see the degradation for yourself.
"""







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
import matplotlib.pyplot as plt
import torch
from dataset import load_mnist
import numpy

dataset,_=load_mnist()
image_tensor, label = dataset[0]
print(image_tensor.shape)
# This loop scans the dataset to find the index of the first occurrence of each label from 0 to 9
target_indices = {}
ftarget_indices = {}
for idx, (_, label) in enumerate(dataset):
    # Ensure label is a standard Python integer
    label_item = label.item() if isinstance(label, torch.Tensor) else int(label)
    
    # Map the actual digit (0-9) to its dataset index
    if label_item not in target_indices:
        target_indices[label_item] = idx
        
    # Stop once we have found one example for each of the 10 digits
    if len(target_indices) == 10:
        break

# Now, target_indices[5] will give you the correct index for the digit 5!
print(target_indices)

img=img_np = image_tensor.squeeze(0).numpy()
plt.imshow(img_np, cmap="gray")
plt.title(f"MNIST Digit: {label}")
plt.axis("off")
plt.show()
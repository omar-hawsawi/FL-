import torch
import torch.nn as nn
import torch.optim as optim
import DLG
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


# 1. Detach and normalize the optimized dummy tensor
reconstructed_img = DLG.dummy_x.detach().cpu().squeeze(0) # Shape: [1, 28, 28]
i=0
# If you want to save it as an image file to look at:
iteration_num=1
transform = ToPILImage()
img = transform(reconstructed_img)
filename = f"{"leaked_reconstruction"}_iter_{iteration_num}.png"
img.save(filename)
print("\nReconstructed image saved as 'leaked_reconstruction.png'!")


# OR if you are running in an environment that supports plots:
plt.imshow(reconstructed_img.squeeze(0), cmap="gray")
plt.title("Reconstructed Leaked Data")
plt.axis("off")
plt.show()



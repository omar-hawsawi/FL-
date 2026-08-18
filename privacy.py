import torch
import math

def clip_and_add_noise(model_weights, initial_weights, max_grad_norm=1.0, noise_multiplier=0.1):
    """
    Applies Differential Privacy to client weight updates (deltas):
    1. Computes the update delta (W_client - W_global).
    2. Clips the update norm to bound client sensitivity.
    3. Adds calibrated Gaussian noise.
    """
    dp_weights = {}
    
    total_norm = 0.0
    for key in model_weights:
        delta = model_weights[key] - initial_weights[key]
        total_norm += delta.pow(2).sum().item()
    total_norm = math.sqrt(total_norm)
    
    clip_factor = max(1.0, total_norm / max_grad_norm)
    
    for key in model_weights:
        delta = model_weights[key] - initial_weights[key]
        clipped_delta = delta / clip_factor
        
        sigma = noise_multiplier * max_grad_norm
        noise = torch.normal(0, sigma, size=clipped_delta.shape, device=clipped_delta.device)
        
        dp_weights[key] = initial_weights[key] + clipped_delta + noise
        
    return dp_weights
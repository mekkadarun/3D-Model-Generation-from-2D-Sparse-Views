"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 12 2025
 * Purpose: Implements specialized loss functions for 3D reconstruction and
            latent space regularization, including binary cross-entropy for
            voxels and KL divergence for VAE.
"""

import torch
import torch.nn.functional as F

def reconstruction_loss(recon_x, x):
    """Binary cross entropy for voxel reconstruction"""
    # Ensure proper dimensions
    if recon_x.dim() == 5 and recon_x.size(1) == 1:
        recon_x = recon_x.squeeze(1)
    if x.dim() == 5 and x.size(1) == 1:
        x = x.squeeze(1)
    
    # BCE loss works better for binary voxels
    loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
    return loss

def kl_divergence_loss(mu, logvar):
    """KL divergence loss with proper scaling"""
    # Proper KL divergence calculation - note the negative sign!
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Scale by batch size
    return kl_loss / mu.size(0)
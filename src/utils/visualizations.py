"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 12 2025
 * Purpose: Implements visualization utilities for Multi-View 3D-VAE-GAN,
            including 2D projections, 3D voxel rendering, and training
            curve plotting for model performance analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from PIL import Image

def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)  # Convert to binary and limit to first 8 samples
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        
    plt.savefig(os.path.join(path, f"{str(iteration).zfill(3)}.png"), bbox_inches='tight')
    plt.close()
    
    # Save raw voxel data
    with open(os.path.join(path, f"{str(iteration).zfill(3)}.npy"), "wb") as f:
        np.save(f, voxels)

def visualize_input_with_poses(images, poses, voxels, output_dir, epoch, threshold=0.1):
    fig, axes = plt.subplots(2, len(images) + 1, figsize=(4*(len(images) + 1), 8))
    
    # Display images in the first row
    for i, img in enumerate(images):
        # Handle potential batch dimension
        if img.dim() == 4:  # If [batch, C, H, W]
            img = img[0]  # Take first item from batch
        
        # Denormalize image if needed
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = np.clip((img_np + 1) / 2, 0, 1)  # Denormalize from [-1,1] to [0,1]
        
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"View {i+1}")
        axes[0, i].axis('off')
        
        # Display pose as text in compact format
        if poses is not None:
            pose = poses[i]
            if pose.dim() > 1:  # If batched
                pose = pose[0]  # Take first item from batch
                
            pose = pose.cpu().numpy()
            R = pose[:9].reshape(3, 3)
            T = pose[9:]
            # Show only key elements of rotation and translation
            pose_text = f"R: [{R[0,0]:.2f} {R[0,1]:.2f} {R[0,2]:.2f}]\n"
            pose_text += f"    [{R[1,0]:.2f} {R[1,1]:.2f} {R[1,2]:.2f}]\n"
            pose_text += f"    [{R[2,0]:.2f} {R[2,1]:.2f} {R[2,2]:.2f}]\n"
            pose_text += f"T: [{T[0]:.2f} {T[1]:.2f} {T[2]:.2f}]"
            axes[1, i].text(0.5, 0.5, pose_text, ha='center', va='center', fontsize=8)
            axes[1, i].axis('off')
    
    # Display target voxel grid
    voxel_np = voxels[0].detach().cpu().numpy()
    if voxel_np.ndim == 4:  # If [1, C, D, H, W]
        voxel_np = voxel_np.squeeze(0)  # Remove batch dimension
    if voxel_np.ndim == 4:  # If [C, D, H, W]
        voxel_np = voxel_np.squeeze(0)  # Remove channel dimension
    
    axes[0, -1].imshow(np.max(voxel_np, axis=0))
    axes[0, -1].set_title("Target Voxel (XY)")
    axes[0, -1].axis('off')
    
    axes[1, -1].imshow(np.max(voxel_np, axis=1))
    axes[1, -1].set_title("Target Voxel (XZ)")
    axes[1, -1].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "input_visualizations"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "input_visualizations", f"input_epoch_{epoch+1}.png"))
    plt.close(fig)

def visualize_results(encoder, generator, images, voxels, epoch, output_dir, threshold=0.1, poses=None):
    with torch.no_grad():
        # Handle voxels dimension
        sample_voxel = voxels[0].detach().cpu().numpy()
        if sample_voxel.ndim == 4:  # If [1, C, D, H, W]
            sample_voxel = sample_voxel.squeeze(0)
        if sample_voxel.ndim == 4:  # If [C, D, H, W]
            sample_voxel = sample_voxel.squeeze(0)
            
        # Generate reconstruction
        if poses is not None:
            z, mu, logvar = encoder(images, poses)
        else:
            z, mu, logvar = encoder(images)
            
        recon_voxel = generator(z)[0].detach().cpu().numpy()
        if recon_voxel.ndim == 4:  # If [1, C, D, H, W]
            recon_voxel = recon_voxel.squeeze(0)
        if recon_voxel.ndim == 4:  # If [C, D, H, W]
            recon_voxel = recon_voxel.squeeze(0)

    # Create the visualization grid - similar to the paper's style
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Display original voxel projections
    axes[0, 0].imshow(np.max(sample_voxel, axis=0), cmap='viridis')
    axes[0, 0].set_title("Original XY")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.max(sample_voxel, axis=1), cmap='viridis')
    axes[0, 1].set_title("Original XZ")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.max(sample_voxel, axis=2), cmap='viridis')
    axes[0, 2].set_title("Original YZ")
    axes[0, 2].axis('off')

    # Display reconstructed voxel projections
    vmin, vmax = 0, 1
    axes[1, 0].imshow(np.max(recon_voxel, axis=0), cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"Recon XY (max: {np.max(recon_voxel):.2f})")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.max(recon_voxel, axis=1), cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f"Recon XZ (thresh: {threshold})")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.max(recon_voxel, axis=2), cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title(f"Recon YZ (voxels: {np.sum(recon_voxel > threshold)})")
    axes[1, 2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "visualizations", f"epoch_{epoch+1}.png"))
    plt.close(fig)
    
    # Save the 3D scatter plot visualization
    SavePloat_Voxels(
        np.concatenate([
            sample_voxel.reshape(1, *sample_voxel.shape), 
            recon_voxel.reshape(1, *recon_voxel.shape)
        ]), 
        os.path.join(output_dir, "visualizations"), 
        f"3d_epoch_{epoch+1}"
    )
    
    # Also visualize input images with poses
    if poses is not None:
        visualize_input_with_poses(images, poses, voxels, output_dir, epoch, threshold)

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    epochs = range(1, len(history['train_D']) + 1)
    
    # Plot GAN losses
    axes[0].plot(epochs, history['train_D'], label='Discriminator')
    axes[0].plot(epochs, history['train_G'], label='Generator')
    axes[0].set_title('GAN Losses')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot encoder losses
    axes[1].plot(epochs, history['train_E'], label='Total')
    axes[1].plot(epochs, history['train_recon'], label='Reconstruction')
    axes[1].plot(epochs, history['train_kl'], label='KL Divergence')
    axes[1].set_title('Encoder Losses')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def test_visualization(images, voxel_grid_np, output_folder, num_views, threshold=0.1):
    n_images = len(images)
    n_projs = 3  # Three projections (XY, XZ, YZ)
    n_cols = max(n_images, n_projs)
    n_rows = 2

    fig = plt.figure(figsize=(4 * n_cols, 8))
    
    # Display input images
    for i in range(n_images):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        img_np = images[i].permute(1, 2, 0).numpy()
        img_np = np.clip((img_np + 1) / 2, 0, 1)  # Denormalize
        ax.imshow(img_np)
        ax.set_title(f"Input View {i+1}")
        ax.axis('off')

    # Display reconstructed voxel projections
    ax = fig.add_subplot(n_rows, n_cols, n_images + 1)
    ax.imshow(np.max(voxel_grid_np, axis=0), cmap='viridis')
    ax.set_title(f"3D Recon XY (max: {np.max(voxel_grid_np):.2f})")
    ax.axis('off')

    ax = fig.add_subplot(n_rows, n_cols, n_images + 2)
    ax.imshow(np.max(voxel_grid_np, axis=1), cmap='viridis')
    ax.set_title(f"3D Recon XZ (voxels: {np.sum(voxel_grid_np > threshold)})")
    ax.axis('off')

    ax = fig.add_subplot(n_rows, n_cols, n_images + 3)
    ax.imshow(np.max(voxel_grid_np, axis=2), cmap='viridis')
    ax.set_title("3D Recon YZ")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "test_reconstruction.png"), dpi=150)
    plt.close(fig)

def test_voxel3d(voxel_grid_np, output_folder, threshold=0.1):
    voxels_binary = voxel_grid_np > threshold
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot only the non-zero voxels for efficiency
    x, y, z = voxels_binary.nonzero()
    ax.scatter(x, y, z, zdir='z', c='red', marker='o', alpha=0.7, s=5)
    
    # Set axis labels and title
    ax.set_title(f"3D Voxel Reconstruction\nVoxels: {np.sum(voxels_binary)}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(os.path.join(output_folder, "3d_voxel_plot.png"), dpi=150)
    
    # Save the voxel data for later use
    np.save(os.path.join(output_folder, "voxel_reconstruction.npy"), voxels_binary)
    
    plt.close(fig)
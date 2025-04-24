"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 15 2025
 * Purpose: Implements testing functionality for Multi-View 3D-VAE-GAN with
            simplified visualization that focuses on 2D projections and basic
            3D scatter visualization.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from PIL import Image
import json
import argparse
import torchvision.transforms as transforms

from model import MultiViewEncoder, Generator

def load_model(model_path, z_size=128, cube_len=32, num_views=4, pose_dim=12, combine_type='mean', device='cuda'):
    encoder = MultiViewEncoder(
        img_size=224,
        z_size=z_size,
        num_views=num_views,
        pose_dim=pose_dim,
        combine_type=combine_type
    ).to(device)
    
    generator = Generator(
        z_size=z_size,
        cube_len=cube_len
    ).to(device)
    
    encoder_path = os.path.join(model_path, "encoder.pth")
    generator_path = os.path.join(model_path, "generator.pth")
    
    if os.path.exists(encoder_path) and os.path.exists(generator_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        generator.load_state_dict(torch.load(generator_path, map_location=device))
        print(f"Successfully loaded models from {model_path}")
        return encoder, generator
    else:
        print(f"Error: Model files not found at {model_path}")
        print(f"Looked for: {encoder_path} and {generator_path}")
        return None, None

def load_test_data(test_dir, num_views=4, image_size=224, use_pose=True, use_masks=True):
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) < num_views:
        print(f"Warning: Found only {len(image_files)} images, but requested {num_views} views")
        print(f"Images found: {image_files}")
        return None, None
    
    image_files.sort()
    selected_images = image_files[:num_views]
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    images = []
    for img_file in selected_images:
        img_path = os.path.join(test_dir, img_file)
        try:
            img = Image.open(img_path).convert('RGB')
            print(f"Loaded image: {img_path}, size: {img.size}")
            
            if use_masks:
                mask_dir = os.path.join(test_dir, "masks")
                if os.path.exists(mask_dir):
                    base_name = os.path.splitext(img_file)[0]
                    mask_candidates = [
                        os.path.join(mask_dir, img_file),
                        os.path.join(mask_dir, f"{base_name}.png"),
                        os.path.join(mask_dir, f"{base_name}.jpg"),
                        os.path.join(mask_dir, f"{base_name}_mask.png"),
                    ]
                    
                    mask_path = None
                    for path in mask_candidates:
                        if os.path.exists(path):
                            mask_path = path
                            break
                    
                    if mask_path:
                        mask = Image.open(mask_path).convert('L')
                        print(f"Loaded mask: {mask_path}, size: {mask.size}")
                        
                        mask_np = np.array(mask) > 128
                        img_np = np.array(img)
                        for c in range(3):
                            img_np[:, :, c] = np.where(mask_np, img_np[:, :, c], 0)
                        img = Image.fromarray(img_np)
                    else:
                        print(f"No matching mask found for {img_file}")
            
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None
    
    if use_pose:
        pose_dir = os.path.join(test_dir, "poses")
        if not os.path.exists(pose_dir):
            print(f"Warning: Pose directory not found at {pose_dir}")
            return images, None
        
        pose_files = [f for f in os.listdir(pose_dir) if f.endswith('.json')]
        pose_files.sort()
        
        if len(pose_files) < num_views:
            print(f"Warning: Found only {len(pose_files)} pose files, but requested {num_views} views")
            print(f"Pose files found: {pose_files}")
            return images, None
        
        selected_poses = pose_files[:num_views]
        poses = []
        
        for pose_file in selected_poses:
            pose_path = os.path.join(pose_dir, pose_file)
            try:
                with open(pose_path, 'r') as f:
                    pose_data = json.load(f)
                R = torch.tensor(pose_data.get("R", np.eye(3))).float().flatten()
                T = torch.tensor(pose_data.get("T", np.zeros(3))).float()
                pose = torch.cat([R, T])
                poses.append(pose)
                print(f"Loaded pose from: {pose_path}")
            except Exception as e:
                print(f"Error loading pose {pose_path}: {e}")
                R = torch.eye(3).flatten()
                T = torch.zeros(3)
                pose = torch.cat([R, T])
                poses.append(pose)
                
        return images, poses
    
    return images, None

def visualize_input_images(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 4, 4))
    
    if n_images == 1:
        axes = [axes]
    
    for i, img_tensor in enumerate(images):
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np + 1) / 2
        img_np = np.clip(img_np, 0, 1)
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"View {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_images.png"), dpi=150)
    plt.close(fig)
    
    print(f"Input image visualization saved to {output_dir}")

def visualize_voxels(voxel_grid_np, output_path, threshold=0.5, dpi=150):
    voxels_binary = voxel_grid_np > threshold
    
    # Create visualization with projections and 3D scatter
    fig = plt.figure(figsize=(15, 7))
    
    # Create 2D projections (maximum intensity)
    ax1 = fig.add_subplot(231)
    ax1.imshow(np.max(voxel_grid_np, axis=0), cmap='viridis')
    ax1.set_title("XY Projection")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(232)
    ax2.imshow(np.max(voxel_grid_np, axis=1), cmap='viridis')
    ax2.set_title("XZ Projection")
    ax2.axis('off')
    
    ax3 = fig.add_subplot(233)
    ax3.imshow(np.max(voxel_grid_np, axis=2), cmap='viridis')
    ax3.set_title("YZ Projection")
    ax3.axis('off')
    
    # 3D visualization
    ax4 = fig.add_subplot(212, projection='3d')
    
    # Get the indices of occupied voxels
    x, y, z = np.where(voxels_binary)
    
    if len(x) == 0:
        print(f"Warning: No voxels above threshold {threshold}")
        ax4.text(0.5, 0.5, 0.5, "No voxels above threshold", 
                ha='center', va='center', transform=ax4.transAxes)
    else:
        # Scatter plot of voxel centers
        ax4.scatter(x, y, z, c='red', marker='o', s=10, alpha=0.7)
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax4.set_xlim(mid_x - max_range, mid_x + max_range)
        ax4.set_ylim(mid_y - max_range, mid_y + max_range)
        ax4.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax4.set_title(f"3D Voxel Visualization (threshold={threshold}, voxel count={np.sum(voxels_binary)})")
    ax4.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    
    print(f"Voxel visualization saved to {output_path}")
    
    # Save raw voxel data
    np.save(f"{os.path.splitext(output_path)[0]}_data.npy", voxels_binary)

def test_multi_view_vaegan(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.test_dir is None and hasattr(args, 'input_dir') and args.input_dir is not None:
        args.test_dir = args.input_dir
    
    encoder, generator = load_model(
        args.model_path, 
        args.z_size,
        args.cube_len,
        args.num_views,
        12 if args.use_pose else None,
        args.combine_type,
        device
    )
    
    if encoder is None or generator is None:
        print("Failed to load models. Exiting.")
        return
    
    images, poses = load_test_data(
        args.test_dir,
        args.num_views,
        args.image_size,
        args.use_pose,
        args.use_masks
    )
    
    if images is None:
        print("Failed to load test data. Exiting.")
        return
    
    visualize_input_images(images, args.output_dir)
    
    batch_images = [img.unsqueeze(0).to(device) for img in images]
    
    if poses is not None and args.use_pose:
        batch_poses = [pose.unsqueeze(0).to(device) for pose in poses]
    else:
        batch_poses = None
    
    with torch.no_grad():
        try:
            if args.use_pose and batch_poses is not None:
                z, _, _ = encoder(batch_images, batch_poses)
            else:
                z, _, _ = encoder(batch_images)
            
            voxel_grid = generator(z)
            print(f"Generated voxel grid with shape: {voxel_grid.shape}")
            print(f"Voxel range: [{voxel_grid.min().item():.4f}, {voxel_grid.max().item():.4f}]")
            
            voxel_grid_np = voxel_grid[0, 0].cpu().numpy()
            
            # Save visualizations with specified threshold
            voxel_vis_path = os.path.join(args.output_dir, "voxel_visualization.png")
            visualize_voxels(voxel_grid_np, voxel_vis_path, threshold=args.threshold)
            
            # Save multi-view visualization (second visualization)
            multi_view_path = os.path.join(args.output_dir, "multi_view.png")
            
            # Create multi-view figure with 8 different views
            fig = plt.figure(figsize=(16, 8))
            gs = gridspec.GridSpec(2, 4)
            gs.update(wspace=0.05, hspace=0.05)
            
            voxels_binary = voxel_grid_np > args.threshold
            x, y, z = voxels_binary.nonzero()
            
            if len(x) == 0:
                print(f"Warning: No voxels above threshold {args.threshold}")
                plt.figtext(0.5, 0.5, "No voxels above threshold", ha='center', fontsize=20)
            else:
                # Set ranges for consistent scaling
                max_range = np.max([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()])
                mid_x = (x.max()+x.min()) * 0.5
                mid_y = (y.max()+y.min()) * 0.5
                mid_z = (z.max()+z.min()) * 0.5
                x_range = (mid_x - max_range*0.5, mid_x + max_range*0.5)
                y_range = (mid_y - max_range*0.5, mid_y + max_range*0.5)
                z_range = (mid_z - max_range*0.5, mid_z + max_range*0.5)
                
                # Draw from 8 different angles
                angles = [
                    (30, 45), (30, 135), (30, 225), (30, 315),
                    (60, 45), (60, 135), (60, 225), (60, 315)
                ]
                
                for i, angle in enumerate(angles):
                    ax = plt.subplot(gs[i], projection='3d')
                    ax.scatter(x, y, z, c='red', s=5, alpha=0.8)
                    
                    # Set view angle
                    ax.view_init(elev=angle[0], azim=angle[1])
                    
                    # Set equal aspect and ranges
                    ax.set_xlim(x_range)
                    ax.set_ylim(y_range)
                    ax.set_zlim(z_range)
                    
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_zticklabels([])
                    ax.set_axis_off()
            
            # Save the figure
            plt.savefig(multi_view_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"Multi-view visualization saved to {multi_view_path}")
            
            # Save raw voxel data as well
            np.save(os.path.join(args.output_dir, "voxel_data.npy"), voxel_grid_np)
            
        except Exception as e:
            print(f"Error during 3D reconstruction: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print(f"Testing completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Multi-View 3D-VAE-GAN with simplified visualization")
    
    # Required parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to test directory containing images and pose files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for test results")
    
    # Alternative name for test_dir
    parser.add_argument("--input_dir", type=str,
                        help="Alternative name for test_dir")
    
    # Model parameters
    parser.add_argument("--z_size", type=int, default=128,
                        help="Latent vector size")
    parser.add_argument("--cube_len", type=int, default=32,
                        help="3D cube length")
    parser.add_argument("--num_views", type=int, default=4,
                        help="Number of views to use")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    
    # Configuration parameters
    parser.add_argument("--use_pose", action="store_true", default=True,
                        help="Whether to use pose information")
    parser.add_argument("--use_masks", action="store_true", default=True,
                        help="Whether to use depth masks (white object, black background)")
    parser.add_argument("--combine_type", type=str, choices=["mean", "max"], default="mean",
                        help="Method for combining multiple views")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for voxel visualization")
    parser.add_argument("--use_cuda", action="store_true", default=True,
                        help="Whether to use CUDA (if available)")
    
    args = parser.parse_args()
    
    # Allow input_dir as an alternative to test_dir
    if args.test_dir is None and args.input_dir is not None:
        args.test_dir = args.input_dir
    
    test_multi_view_vaegan(args)
"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 10 2025
 * Purpose: Implements training functionality for Multi-View 3D-VAE-GAN model, 
            supporting multiple views, pose information, and resumable training 
            from checkpoints.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import argparse
import yaml
import time
from datetime import datetime

from model import MultiViewEncoder, Generator, Discriminator
from data.dataset import CO3DMultiViewDataset as CO3DDataset
from utils.visualizations import visualize_results, plot_training_curves

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def save_checkpoint(epoch, encoder, generator, discriminator, 
                   encoder_optimizer, generator_optimizer, discriminator_optimizer, 
                   checkpoint_dir, history=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, "encoder.pth"))
    torch.save(generator.state_dict(), os.path.join(checkpoint_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, "discriminator.pth"))
    
    torch.save(encoder_optimizer.state_dict(), os.path.join(checkpoint_dir, "encoder_optimizer.pth"))
    torch.save(generator_optimizer.state_dict(), os.path.join(checkpoint_dir, "generator_optimizer.pth"))
    torch.save(discriminator_optimizer.state_dict(), os.path.join(checkpoint_dir, "discriminator_optimizer.pth"))
    
    if history is not None:
        training_info = {
            'epoch': epoch,
            'history': history
        }
        torch.save(training_info, os.path.join(checkpoint_dir, "training_info.pth"))
    
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(checkpoint_path, encoder, generator, discriminator, 
                   encoder_optimizer=None, generator_optimizer=None, discriminator_optimizer=None):
    try:
        # Load models
        encoder.load_state_dict(torch.load(os.path.join(checkpoint_path, "encoder.pth")))
        generator.load_state_dict(torch.load(os.path.join(checkpoint_path, "generator.pth")))
        discriminator.load_state_dict(torch.load(os.path.join(checkpoint_path, "discriminator.pth")))
        
        print(f"Model weights loaded successfully from {checkpoint_path}")
        
        start_epoch = 0
        history = {
            'train_D': [], 'train_G': [], 'train_E': [],
            'train_recon': [], 'train_kl': []
        }
        
        # Try to load optimizer states if provided
        if encoder_optimizer and generator_optimizer and discriminator_optimizer:
            try:
                encoder_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "encoder_optimizer.pth")))
                generator_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "generator_optimizer.pth")))
                discriminator_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "discriminator_optimizer.pth")))
                print("Optimizer states loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load optimizer states: {e}")
        
        # Try to load training info
        training_info_path = os.path.join(checkpoint_path, "training_info.pth")
            
        if os.path.exists(training_info_path):
            try:
                training_info = torch.load(training_info_path)
                history = training_info.get('history', history)
                start_epoch = training_info.get('epoch', 0) + 1
                print(f"Resuming from epoch {start_epoch}")
            except Exception as e:
                print(f"Error loading training history: {e}")
        
        return start_epoch, history
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise e

def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

def train_multiview_vaegan(config, resume_from=None, start_epoch_override=None):
    print(f"Starting Multi-View 3D-VAE-GAN training with:")
    print(f"  - {config.batch_size} batch size")
    print(f"  - {config.num_views} views")
    print(f"  - {config.combine_type} pooling type")
    
    if hasattr(config, 'seed'):
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir
    if hasattr(config, 'experiment_name') and config.experiment_name:
        output_subdir = config.experiment_name
    else:
        output_subdir = f"mv_{config.num_views}views_{config.combine_type}"
        if config.use_pose:
            output_subdir += "_with_pose"
        else:
            output_subdir += "_without_pose"
        
        if resume_from:
            output_subdir += f"_continued_{timestamp}"
        else:
            output_subdir += f"_{timestamp}"
    
    output_dir = os.path.join(output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(vars(config), f)
    
    dataset = CO3DDataset(
        root_dir=config.input_dir,
        obj=config.obj,
        num_views=config.num_views,
        image_size=config.image_size,
        apply_mask=config.apply_mask,
        use_pose=config.use_pose
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=dataset.get_collate_fn(),
        drop_last=True,
        pin_memory=True
    )
    
    print(f"Dataset loaded with {len(dataset)} sequences")

    pose_dim = 12 if config.use_pose else None
    
    encoder = MultiViewEncoder(
        img_size=config.image_size,
        z_size=config.z_size,
        num_views=config.num_views,
        pose_dim=pose_dim,
        combine_type=config.combine_type
    ).to(device)
    
    generator = Generator(
        z_size=config.z_size,
        cube_len=config.cube_len
    ).to(device)
    
    discriminator = Discriminator(
        cube_len=config.cube_len
    ).to(device)

    # Initialize optimizers first (we might update their state later)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.e_lr, betas=(0.5, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(), lr=config.g_lr, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(0.5, 0.999))

    # Initialize training history and epoch counter
    start_epoch = 0
    history = {
        'train_D': [], 'train_G': [], 'train_E': [],
        'train_recon': [], 'train_kl': []
    }
    
    # Load checkpoint if provided
    if resume_from:
        try:
            start_epoch, history = load_checkpoint(
                resume_from, encoder, generator, discriminator, 
                encoder_optimizer, generator_optimizer, discriminator_optimizer
            )
            print(f"Successfully loaded checkpoint from {resume_from}")
        except Exception as e:
            print(f"Failed to load complete checkpoint: {e}")
            print("Will start training from scratch")
            start_epoch = 0
    
    # Override start epoch if specified
    if start_epoch_override is not None:
        start_epoch = start_epoch_override
        print(f"Starting epoch overridden to: {start_epoch}")
    
    # Fill in history with None values if resuming to maintain proper indexing
    for key in history:
        while len(history[key]) < start_epoch:
            history[key].append(None)
    
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    recon_weight = getattr(config, 'recon_weight', 50.0)
    
    kl_weight_start = getattr(config, 'kl_weight_start', 0.0001)
    kl_weight_end = getattr(config, 'kl_weight_end', 0.01)
    
    print(f"Starting training from epoch {start_epoch} to {config.n_epochs}")
    print(f"Reconstruction weight: {recon_weight}")
    print(f"KL weight: {kl_weight_start} to {kl_weight_end}")

    for epoch in range(start_epoch, config.n_epochs):
        epoch_start_time = time.time()
        
        encoder.train()
        generator.train()
        discriminator.train()
        
        kl_weight = kl_weight_start + (kl_weight_end - kl_weight_start) * min(1.0, epoch / (config.n_epochs * 0.7))
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_e_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        batch_count = 0

        for batch in data_loader:
            if config.use_pose:
                images, voxels, poses = batch
                poses = [p.to(device) for p in poses]
            else:
                images, voxels = batch
                poses = None
                
            batch_size = voxels.size(0)
            voxels = voxels.to(device)
            images = [img.to(device) for img in images]
            
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            discriminator_optimizer.zero_grad()
            
            d_real = discriminator(voxels)
            d_real_loss = bce_loss(d_real, real_labels)
            
            z_random = torch.randn(batch_size, config.z_size).to(device)
            fake_voxels = generator(z_random).detach()
            d_fake = discriminator(fake_voxels)
            d_fake_loss = bce_loss(d_fake, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            discriminator_optimizer.step()
            
            # Train Generator (GAN pathway)
            generator_optimizer.zero_grad()
            z_random = torch.randn(batch_size, config.z_size).to(device)
            fake_voxels = generator(z_random)
            d_fake = discriminator(fake_voxels)
            g_loss_gan = bce_loss(d_fake, real_labels)
            
            # Train Encoder-Generator (VAE pathway)
            encoder_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            
            if config.use_pose:
                z_enc, mu, logvar = encoder(images, poses)
            else:
                z_enc, mu, logvar = encoder(images)
                
            recon_voxels = generator(z_enc)
            
            recon_loss = mse_loss(recon_voxels, voxels)
            kl_loss = kl_divergence_loss(mu, logvar)
            
            # Combined losses and backward
            g_loss_recon = recon_weight * recon_loss
            g_loss = g_loss_gan + g_loss_recon
            g_loss.backward(retain_graph=True)
            generator_optimizer.step()
            
            e_loss = recon_weight * recon_loss + kl_weight * kl_loss
            e_loss.backward()
            encoder_optimizer.step()
            
            # Update metrics
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_e_loss += e_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            batch_count += 1

        # End of epoch processing
        if batch_count > 0:
            epoch_d_loss /= batch_count
            epoch_g_loss /= batch_count
            epoch_e_loss /= batch_count
            epoch_recon_loss /= batch_count
            epoch_kl_loss /= batch_count

        # Record history
        history['train_D'].append(epoch_d_loss)
        history['train_G'].append(epoch_g_loss)
        history['train_E'].append(epoch_e_loss)
        history['train_recon'].append(epoch_recon_loss)
        history['train_kl'].append(epoch_kl_loss)

        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{config.n_epochs} | "
              f"D: {epoch_d_loss:.4f}, G: {epoch_g_loss:.4f}, "
              f"E: {epoch_e_loss:.4f} (Recon: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.4f}) | "
              f"Time: {epoch_time:.2f}s")

        # Save checkpoints periodically
        if (epoch+1) % config.save_freq == 0 or (epoch+1) == config.n_epochs:
            epoch_checkpoint_dir = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
            save_checkpoint(
                epoch, encoder, generator, discriminator,
                encoder_optimizer, generator_optimizer, discriminator_optimizer,
                epoch_checkpoint_dir, history
            )
            
            latest_checkpoint_dir = os.path.join(checkpoint_dir, "latest")
            save_checkpoint(
                epoch, encoder, generator, discriminator,
                encoder_optimizer, generator_optimizer, discriminator_optimizer,
                latest_checkpoint_dir, history
            )

        # Visualize results periodically
        if (epoch+1) % config.vis_freq == 0:
            try:
                with torch.no_grad():
                    vis_images = images
                    vis_voxels = voxels
                    
                    if config.use_pose:
                        vis_poses = poses
                        visualize_results(encoder, generator, vis_images, vis_voxels, epoch, vis_dir, poses=vis_poses)
                    else:
                        visualize_results(encoder, generator, vis_images, vis_voxels, epoch, vis_dir)
                        
                    filtered_history = {
                        'train_D': [x for x in history['train_D'] if x is not None],
                        'train_G': [x for x in history['train_G'] if x is not None],
                        'train_E': [x for x in history['train_E'] if x is not None],
                        'train_recon': [x for x in history['train_recon'] if x is not None],
                        'train_kl': [x for x in history['train_kl'] if x is not None]
                    }
                    plot_path = os.path.join(vis_dir, f"training_curves_epoch_{epoch+1}.png")
                    plot_training_curves(filtered_history, plot_path)
                    
            except Exception as e:
                print(f"Visualization error at epoch {epoch+1}: {e}")
                import traceback
                traceback.print_exc()

    # Save final models
    final_checkpoint_dir = os.path.join(checkpoint_dir, "final")
    save_checkpoint(
        config.n_epochs-1, encoder, generator, discriminator,
        encoder_optimizer, generator_optimizer, discriminator_optimizer,
        final_checkpoint_dir, history
    )
    
    torch.save(encoder.state_dict(), os.path.join(output_dir, "encoder_final.pth"))
    torch.save(generator.state_dict(), os.path.join(output_dir, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, "discriminator_final.pth"))
    
    filtered_history = {
        'train_D': [x for x in history['train_D'] if x is not None],
        'train_G': [x for x in history['train_G'] if x is not None],
        'train_E': [x for x in history['train_E'] if x is not None],
        'train_recon': [x for x in history['train_recon'] if x is not None],
        'train_kl': [x for x in history['train_kl'] if x is not None]
    }
    plot_path = os.path.join(output_dir, "training_history.png")
    plot_training_curves(filtered_history, plot_path)
    
    print(f"Training completed. Models saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-View 3D-VAE-GAN Training Script")
    
    parser.add_argument("--config", type=str, default="config/pose.yaml", 
                        help="Path to config file")
    
    parser.add_argument("--input_dir", type=str, 
                        help="Input directory containing CO3D data")
    parser.add_argument("--output_dir", type=str, 
                        help="Output directory for models and visualizations")
    
    parser.add_argument("--batch_size", type=int, 
                        help="Batch size for training")
    parser.add_argument("--num_views", type=int, 
                        help="Number of views to use")
    parser.add_argument("--image_size", type=int, 
                        help="Size of input images")
    parser.add_argument("--combine_type", type=str, choices=["mean", "max"], 
                        help="Method for combining multiple views")
    parser.add_argument("--use_pose", type=str2bool, 
                        help="Whether to use pose information")
    
    parser.add_argument("--n_epochs", type=int, 
                        help="Number of training epochs")
    parser.add_argument("--g_lr", type=float, 
                        help="Generator learning rate")
    parser.add_argument("--e_lr", type=float, 
                        help="Encoder learning rate")
    parser.add_argument("--d_lr", type=float, 
                        help="Discriminator learning rate")
    parser.add_argument("--recon_weight", type=float, 
                        help="Weight for reconstruction loss")
    parser.add_argument("--kl_weight_start", type=float, 
                        help="Initial KL divergence weight")
    parser.add_argument("--kl_weight_end", type=float, 
                        help="Final KL divergence weight")
    
    # Resume training options
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory for resuming training")
    parser.add_argument("--start_epoch", type=int, default=None,
                        help="Override starting epoch when resuming (specify the epoch to continue from)")
    
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for this experiment run")
    
    args = parser.parse_args()
    
    config_dict = load_config(args.config)
    config = Config(config_dict)
    
    for arg in vars(args):
        if getattr(args, arg) is not None and arg not in ['resume', 'start_epoch']:
            setattr(config, arg, getattr(args, arg))
    
    defaults = {
        'num_workers': 4,
        'save_freq': 10,
        'vis_freq': 5,
        'z_size': 128,
        'cube_len': 32,
        'e_lr': 0.0001,
        'g_lr': 0.0001,
        'd_lr': 0.00001,
        'apply_mask': True,
        'n_epochs': 1000,
        'combine_type': 'mean',
        'num_views': 4,
        'recon_weight': 50.0,
        'kl_weight_start': 0.0001,
        'kl_weight_end': 0.01,
    }
    
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    
    train_multiview_vaegan(config, args.resume, args.start_epoch)
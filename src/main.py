"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 13 2025
 * Purpose: Provides the main entry point for the Multi-View 3D-VAE-GAN system,
            coordinating between training and testing modes with consistent
            configuration handling across different operational modes.
"""

import os
import argparse
import yaml
import torch
from train import train_multiview_vaegan
from test import test_multi_view_vaegan

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-View 3D-VAE-GAN Training/Testing")
    
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", 
                        help="Operation mode (train or test)")
    
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
    parser.add_argument("--image_size", type=int, default=224,
                        help="Size of input images")
    parser.add_argument("--combine_type", type=str, choices=["mean", "max"], default="mean",
                        help="Method for combining multiple views")
    parser.add_argument("--use_pose", type=str2bool, nargs='?', const=True, default=True, 
                        help="Whether to use pose information")
    
    parser.add_argument("--n_epochs", type=int, 
                        help="Number of training epochs")
    parser.add_argument("--g_lr", type=float, 
                        help="Generator learning rate")
    parser.add_argument("--e_lr", type=float, 
                        help="Encoder learning rate")
    parser.add_argument("--d_lr", type=float, 
                        help="Discriminator learning rate")
    
    parser.add_argument("--model_path", type=str, 
                        help="Path to pretrained model for testing")
    parser.add_argument("--test_output", type=str, 
                        help="Output directory for test results")
    
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

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

def main():
    args = parse_args()
    
    config_dict = load_config(args.config)
    config = Config(config_dict)
    
    for arg in vars(args):
        if getattr(args, arg) is not None:
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
        'n_epochs': 100,
        'combine_type': 'mean',
        'num_views': 4
    }
    
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    
    if hasattr(config, 'output_dir'):
        output_subdir = f"mv_{config.num_views}views_{config.combine_type}"
        if config.use_pose:
            output_subdir += "_with_pose"
        else:
            output_subdir += "_without_pose"
        
        config.output_dir = os.path.join(config.output_dir, output_subdir)
        os.makedirs(config.output_dir, exist_ok=True)
    
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    if args.mode == "train":
        print(f"Starting training with {config.num_views} views and {config.combine_type} pooling...")
        train_multiview_vaegan(config)
    else:
        print(f"Starting testing with {config.num_views} views and {config.combine_type} pooling...")
        test_multi_view_vaegan(config)

if __name__ == "__main__":
    main()
"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 8 2025
 * Purpose: Implements dataset loading and preprocessing for Multi-View 3D-VAE-GAN
            with support for CO3D dataset format, including image loading, mask
            application, and pose extraction.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
import data.binvox_rw as binvox_rw

class CO3DMultiViewDataset(Dataset):
    def __init__(self, root_dir, obj='chair', num_views=4, image_size=224, apply_mask=True, use_pose=True, combine_type='mean'):
        self.root_dir = os.path.join(root_dir, obj)
        self.num_views = num_views
        self.image_size = image_size
        self.apply_mask = apply_mask
        self.use_pose = use_pose
        self.combine_type = combine_type
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.sequences = self._find_valid_sequences()
        print(f"Found {len(self.sequences)} valid sequences with at least {num_views} views")

    def _find_valid_sequences(self):
        sequences = []
        for item in os.listdir(self.root_dir):
            if not item.endswith('_dataset_instance'):
                continue
                
            seq_name = item.replace('_dataset_instance', '')
            
            instance_dir = os.path.join(self.root_dir, item)
            processed_dir = os.path.join(self.root_dir, f"{seq_name}_processed")
            
            if not os.path.exists(processed_dir):
                print(f"Skipping {seq_name}: No processed directory found")
                continue
                
            binvox_path = None
            for file in os.listdir(instance_dir):
                if file.endswith('.binvox'):
                    binvox_path = os.path.join(instance_dir, file)
                    break
                    
            if not binvox_path:
                print(f"Skipping {seq_name}: No binvox file found")
                continue
                
            poses_dir = os.path.join(processed_dir, "poses")
            if self.use_pose and not os.path.exists(poses_dir):
                print(f"Skipping {seq_name}: No poses directory found")
                continue
                
            meta_path = os.path.join(processed_dir, "sequence_meta.json")
            if not os.path.exists(meta_path):
                print(f"Skipping {seq_name}: No sequence_meta.json found")
                continue
            
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            frames = metadata.get("frames", [])
            if len(frames) < self.num_views:
                print(f"Skipping {seq_name}: Not enough frames ({len(frames)})")
                continue
            
            try:
                with open(binvox_path, 'rb') as f:
                    voxel_model = binvox_rw.read_as_3d_array(f)
                    voxel_data = voxel_model.data
                    
                if np.sum(voxel_data > 0) == 0:
                    print(f"Skipping {seq_name}: Empty voxel data")
                    continue
                    
                sequences.append({
                    "name": seq_name,
                    "instance_dir": instance_dir,
                    "processed_dir": processed_dir,
                    "binvox_path": binvox_path,
                    "meta_path": meta_path,
                    "num_frames": len(frames)
                })
                
            except Exception as e:
                print(f"Error loading binvox for {seq_name}: {e}")
                
        return sequences
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        try:
            with open(sequence["binvox_path"], 'rb') as f:
                voxel_model = binvox_rw.read_as_3d_array(f)
                voxel_data = voxel_model.data.astype(np.float32)
                voxel_tensor = torch.FloatTensor(voxel_data).unsqueeze(0)
            
            with open(sequence["meta_path"], 'r') as f:
                metadata = json.load(f)
            
            frames = metadata.get("frames", [])
            
            indices = np.linspace(0, len(frames)-1, self.num_views, dtype=int)
            selected_frames = [frames[i] for i in indices]
            
            images = []
            poses = []
            
            for frame in selected_frames:
                img_path = os.path.join(sequence["instance_dir"], frame.get("image", ""))
                
                img_filename = os.path.basename(img_path)
                frame_num = None
                
                if img_filename.startswith("frame"):
                    frame_num = img_filename.replace("frame", "").split(".")[0]
                
                if not os.path.exists(img_path):
                    raise ValueError(f"Image file not found: {img_path}")
                    
                img = Image.open(img_path).convert('RGB')
                    
                if self.apply_mask and "mask" in frame:
                    mask_path = os.path.join(sequence["instance_dir"], frame["mask"])
                    if os.path.exists(mask_path):
                        mask = Image.open(mask_path).convert('L')
                        mask_np = np.array(mask) > 128
                        img_np = np.array(img)
                        for c in range(3):
                            img_np[:, :, c] = np.where(mask_np, img_np[:, :, c], 0)
                        img = Image.fromarray(img_np)
                
                img_tensor = self.transform(img)
                images.append(img_tensor)
                
                if self.use_pose:
                    pose_found = False
                    
                    if frame_num:
                        pose_path = os.path.join(sequence["processed_dir"], "poses", f"frame{frame_num}_pose.json")
                        
                        if os.path.exists(pose_path):
                            with open(pose_path, 'r') as f:
                                pose_data = json.load(f)
                            R = torch.tensor(pose_data.get("R", np.eye(3))).float().flatten()
                            T = torch.tensor(pose_data.get("T", np.zeros(3))).float()
                            pose = torch.cat([R, T])
                            poses.append(pose)
                            pose_found = True
                    
                    if not pose_found:
                        R = torch.eye(3).flatten()
                        T = torch.zeros(3)
                        pose = torch.cat([R, T])
                        poses.append(pose)
            
            if self.use_pose:
                return images, voxel_tensor, poses
            else:
                return images, voxel_tensor
                
        except Exception as e:
            print(f"Error loading sequence {sequence['name']}: {e}")
            default_image = torch.zeros(3, self.image_size, self.image_size)
            default_voxel = torch.zeros(1, 32, 32, 32)
            default_pose = torch.cat([torch.eye(3).flatten(), torch.zeros(3)]).float()
            
            if self.use_pose:
                return [default_image] * self.num_views, default_voxel, [default_pose] * self.num_views
            else:
                return [default_image] * self.num_views, default_voxel

    def get_collate_fn(self):
        def collate_fn(batch):
            if self.use_pose:
                images_batch = [item[0] for item in batch]
                voxels_batch = [item[1] for item in batch]
                poses_batch = [item[2] for item in batch]
                
                stacked_images = []
                stacked_poses = []
                for view_idx in range(self.num_views):
                    view_images = [images[view_idx] for images in images_batch]
                    view_poses = [poses[view_idx] for poses in poses_batch]
                    stacked_images.append(torch.stack(view_images))
                    stacked_poses.append(torch.stack(view_poses))
                
                stacked_voxels = torch.stack(voxels_batch)
                
                return stacked_images, stacked_voxels, stacked_poses
            else:
                images_batch = [item[0] for item in batch]
                voxels_batch = [item[1] for item in batch]
                
                stacked_images = []
                for view_idx in range(self.num_views):
                    view_images = [images[view_idx] for images in images_batch]
                    stacked_images.append(torch.stack(view_images))
                    
                stacked_voxels = torch.stack(voxels_batch)
                
                return stacked_images, stacked_voxels
                
        return collate_fn
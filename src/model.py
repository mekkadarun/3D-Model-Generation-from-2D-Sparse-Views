"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 9 2025
 * Purpose: Implements neural network models for 3D-VAE-GAN architecture,
            including multi-view encoder, 3D generator, and discriminator
            with support for different view aggregation methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEncoder(nn.Module):
    def __init__(self, img_size=224, z_size=128, pose_dim=None):
        super(SimpleEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.pose_dim = pose_dim
        
        self.fc_size = 512 * 7 * 7
        
        if pose_dim:
            self.pose_encoder = nn.Sequential(
                nn.Linear(pose_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            self.fc_mu = nn.Linear(self.fc_size + 64, z_size)
            self.fc_logvar = nn.Linear(self.fc_size + 64, z_size)
        else:
            self.fc_mu = nn.Linear(self.fc_size, z_size)
            self.fc_logvar = nn.Linear(self.fc_size, z_size)

    def forward(self, x, pose=None):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        
        if self.pose_dim and pose is not None:
            pose_feat = self.pose_encoder(pose)
            combined = torch.cat([x, pose_feat], dim=1)
            mu = self.fc_mu(combined)
            logvar = self.fc_logvar(combined)
        else:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            
        return mu, logvar

class MultiViewEncoder(nn.Module):
    def __init__(self, img_size=224, z_size=128, num_views=4, pose_dim=None, combine_type='mean'):
        super(MultiViewEncoder, self).__init__()
        self.base_encoder = SimpleEncoder(img_size, z_size, pose_dim)
        self.z_size = z_size
        self.num_views = num_views
        self.pose_dim = pose_dim
        self.combine_type = combine_type
        
        self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def combine_features(self, features):
        if self.combine_type == 'mean':
            return torch.mean(features, dim=0)
        elif self.combine_type == 'max':
            return torch.max(features, dim=0)[0]
        else:
            raise ValueError(f"Unsupported combine type: {self.combine_type}")

    def forward(self, views, poses=None):
        view_mus, view_logvars, view_zs = [], [], []
        
        for i, view in enumerate(views):
            if self.pose_dim and poses is not None:
                mu, logvar = self.base_encoder(view, poses[i])
            else:
                mu, logvar = self.base_encoder(view)
                
            z = self.reparameterize(mu, logvar)
            
            view_mus.append(mu)
            view_logvars.append(logvar)
            view_zs.append(z)
        
        stacked_mus = torch.stack(view_mus, dim=0)
        stacked_logvars = torch.stack(view_logvars, dim=0)
        stacked_zs = torch.stack(view_zs, dim=0)
        
        weights = F.softmax(self.view_weights, dim=0).view(-1, 1, 1)
        
        combined_z = self.combine_features(stacked_zs)
        
        return combined_z, stacked_mus, stacked_logvars

class Generator(nn.Module):
    def __init__(self, z_size=128, cube_len=32):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.cube_len = cube_len
        
        self.fc = nn.Linear(z_size, 512 * 4 * 4 * 4)
        
        self.block1_bn = nn.BatchNorm3d(512)
        self.block1_relu = nn.ReLU(True)
        self.block1_conv = nn.ConvTranspose3d(512, 256, 4, 2, 1)
        self.block2_bn = nn.BatchNorm3d(256)
        self.block2_relu = nn.ReLU(True)
        self.block2_conv = nn.ConvTranspose3d(256, 128, 4, 2, 1)
        self.block3_bn = nn.BatchNorm3d(128)
        self.block3_relu = nn.ReLU(True)
        self.block3_conv = nn.ConvTranspose3d(128, 64, 4, 2, 1)
        self.final_bn = nn.BatchNorm3d(64)
        self.final_relu = nn.ReLU(True)
        self.final_conv = nn.ConvTranspose3d(64, 1, 3, 1, 1)
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4, 4)
        
        x = self.block1_bn(x)
        x = self.block1_relu(x)
        x = self.block1_conv(x)
        
        x = self.block2_bn(x)
        x = self.block2_relu(x)
        x = self.block2_conv(x)
        
        x = self.block3_bn(x)
        x = self.block3_relu(x)
        x = self.block3_conv(x)
        
        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.final_conv(x)
        
        x = self.final_act(x) * 1.2
        x = torch.clamp(x, 0, 1)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, cube_len=32):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(128, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return x.view(-1, 1)
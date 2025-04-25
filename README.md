# 3D Model Generation from 2D Sparse Views

This repository implements a Multi-View 3D-VAE-GAN architecture for generating 3D shapes from 2D images. The model supports both single-view and multi-view training, with optional pose encoding for improved 3D reconstruction.

## Features

- **Multi-view reconstruction**: Generate 3D models from multiple 2D images
- **Pose encoding**: Optional camera pose integration for improved 3D alignment
- **Flexible architecture**: Configurable latent space size, voxel resolution, and training parameters
- **Resumable training**: Continue training from any checkpoint
- **Visualization suite**: Comprehensive visualization tools for both training and evaluation

## Project Structure

```
├── config/
│   └── pose.yaml                     # Configuration for model with pose encoding
├── src
│    ├── data/
│    │   └── dataset.py               # CO3D dataset loading and preprocessing
│    │   └── binvox_rw.py             # Binary voxel file I/O utilities
│    ├── preprocessing/   
│    │   └── co3d_ply2voxel.py        # Point cloud to voxel conversion utility
│    │   └── extract_poses.py         # Camera pose extraction from 
│    │   └── preprocess_data.py       # Data organization and metadata creation
│    │   └── preprocess_sequence.py   # Preprocessing pipeline coordinator
│    │   └── split_data.py            # Dataset splitting into train/test/validation
│    ├── utils/
│    │   └── visualizations.py        # Visualization utilities
│    │   └── losses.py                # Custom loss functions
│    ├── main.py                      # Main execution script
│    ├── model.py                     # Neural network architecture definitions
│    ├── train.py                     # Training loop implementation
│    ├── test.py                      # Testing and visualization functionality
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- PIL
- PyYAML
- torchvision

## Data Preparation

The system is designed to work with the Common Objects in 3D (CO3D) dataset. The expected directory structure is:

```
input/
└── category/                       # Category folder (e.g., "chair")
    ├── sequence_name_processed/    # Processed data
    │   ├── sequence_meta.json      # Metadata file
    │   └── poses/                  # Camera poses folder
    └── sequence_name_dataset_instance/  # Dataset instance folder
        ├── depths/                 # Depth images
        ├── depth_masks/            # Depth masks
        ├── images/                 # RGB images
        ├── masks/                  # Segmentation masks
        └── sequence_name.binvox    # Binvox voxel representation
```

## Training

The model supports several training configurations:

### Basic Training

```bash
python train.py --config config/pose.yaml --input_dir <path_to_chair> --output_dir models/experiments
```

### Multi-View Training with Pose

```bash
python3 train.py --config ../config/pose.yaml --input_dir <path_to_chair> --output_dir <path_to_output> --batch_size 8 --num_views 4 --combine_type mean --use_pose True --n_epochs 500

```

### Resuming Training

```bash
# Resume from latest checkpoint
python3 train.py --config ../config/pose.yaml --input_dir <path_to_chair> --output_dir <path_to_output>


# Resume from specific epoch
python3 train.py --config ../config/pose.yaml --input_dir <path_to_chair> --output_dir <path_to_output> --resume models/experiments/checkpoints/epoch_200

# Resume with epoch override
python3 train.py --config ../config/pose.yaml --input_dir <path_to_chair> --output_dir <path_to_output> --resume models/experiments/checkpoints/latest --start_epoch 300
```

## Testing

```bash
python3 test.py --model_path ../final_model/checkpoints/epoch_1000/ --test_dir ../test/chair/ --output_dir <path_to_output> --num_views 4 --use_pose --threshold 0.4

```

## Configuration Parameters

Key parameters in the configuration files include:

### Model Parameters
- `n_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `z_size`: Latent vector dimension
- `cube_len`: 3D voxel grid resolution
- `image_size`: Input image size

### Training Parameters
- `obj`: Object category (e.g., 'chair')
- `num_views`: Number of views per object
- `combine_type`: Method for combining views ('mean' or 'max')
- `use_pose`: Whether to use pose encoding

### Optimization Parameters
- `g_lr`, `e_lr`, `d_lr`: Learning rates for generator, encoder, and discriminator
- `recon_weight`: Weight for reconstruction loss
- `kl_weight_start`, `kl_weight_end`: KL divergence annealing parameters

## Model Architecture

The model consists of three main components:

1. **Encoder**: Processes 2D images into a latent space
   - Single-view encoder for individual images
   - Multi-view encoder for combining information from multiple viewpoints
   - Optional pose encoding for camera position awareness

2. **Generator**: Converts latent vectors into 3D voxel grids
   - Transposed 3D convolutions for upsampling
   - Generates a binary voxel occupancy grid

3. **Discriminator**: Classifies real vs. generated 3D models
   - 3D convolutional architecture
   - Trained adversarially against the generator

## Troubleshooting

### Empty Voxel Visualizations
If voxel visualizations appear empty during testing:
- Try lowering the threshold value with `--threshold 0.1`
- Check if the model has been trained for enough epochs

### CUDA Out of Memory
If you encounter memory issues:
- Reduce batch size
- Use fewer views in multi-view mode
- Reduce voxel resolution (`cube_len`)

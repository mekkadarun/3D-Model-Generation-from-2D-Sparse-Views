import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import binvox_rw
import argparse
from pathlib import Path


class CO3D_to_Voxel_Pipeline:
    def __init__(self, dataset_root, categories=None, voxel_resolution=32):
        """
        Pipeline to convert CO3D data to voxel representations.

        Args:
            dataset_root: Root directory of CO3D dataset
            categories: List of categories to process (None means all)
            voxel_resolution: Resolution of output voxel grid (default: 32³)
        """
        self.dataset_root = Path(dataset_root)
        self.categories = categories
        self.voxel_resolution = voxel_resolution

    def run(self):
        """Run the complete pipeline"""
        if self.categories is None:
            # Process all categories found in the dataset root
            self.categories = [d for d in os.listdir(self.dataset_root)
                               if os.path.isdir(os.path.join(self.dataset_root, d))]

        print(f"Processing categories: {self.categories}")
        # Store the first processed file as a sample
        sample_instance_path = None
        for category in self.categories:
            print(f"Processing category: {category}")


            # Find all sequence base names (without _processed or _dataset_instance)
            all_dirs = [d for d in os.listdir(os.path.join(self.dataset_root, category))
                        if os.path.isdir(os.path.join(self.dataset_root, category, d))]

            # Extract base names that have both processed and dataset_instance versions
            base_sequences = set()
            processed_dirs = set()
            dataset_instance_dirs = set()

            # First, collect all the processed and dataset_instance directories
            for d in all_dirs:
                if "_processed" in d:
                    processed_dirs.add(d.replace("_processed", ""))
                elif "_dataset_instance" in d:
                    dataset_instance_dirs.add(d.replace("_dataset_instance", ""))

            # Then, only keep base names that appear in both sets
            base_sequences = processed_dirs.intersection(dataset_instance_dirs)

            # Process each base sequence
            for base_seq in tqdm(base_sequences, desc=f"Processing {category} sequences"):
                processed_dir = os.path.join(self.dataset_root, category, f"{base_seq}_processed")
                dataset_instance_dir = os.path.join(self.dataset_root, category, f"{base_seq}_dataset_instance")
                # Output to the same directory as the point cloud file
                pointcloud_path = os.path.join(dataset_instance_dir, "pointcloud.ply")
                if os.path.exists(pointcloud_path):
                    # Create output filename in the dataset_instance directory
                    output_file = os.path.join(dataset_instance_dir, f"{base_seq}.binvox")
                    self.process_sequence_pair(processed_dir, dataset_instance_dir, output_file)
                    # Store the first processed file as a sample
                    if sample_instance_path is None:
                        sample_instance_path = dataset_instance_dir
                else:
                    print(f"No pointcloud file found for {base_seq}, skipping.")
        return sample_instance_path

    def process_sequence_pair(self, processed_dir, dataset_instance_dir, output_path):
        """Process a pair of processed and dataset_instance directories to create a voxel representation"""
        try:
            # First try to use point cloud file from dataset_instance directory
            pointcloud_path = os.path.join(dataset_instance_dir, "pointcloud.ply")
            if os.path.exists(pointcloud_path):
                # Use the provided point cloud
                pointcloud = self.load_pointcloud_from_ply(pointcloud_path)
            else:
                print(f"No pointcloud file found for {pointcloud_path}, skipping.")

            # Convert point cloud to voxel grid
            voxel_grid = self.pointcloud_to_voxel(pointcloud)

            # Save voxel grid in binvox format
            self.save_as_binvox(voxel_grid, output_path)

        except Exception as e:
            print(f"Error processing sequence {processed_dir}: {e}")

    def load_pointcloud_from_ply(self, pointcloud_path):
        """Load a point cloud from a PLY file"""
        pcd = o3d.io.read_point_cloud(pointcloud_path)
        print(pcd)
        return np.asarray(pcd.points)

    def pointcloud_to_voxel(self, points, pad_factor=0.05):
        """Convert point cloud to voxel grid"""
        if len(points) == 0:
            raise ValueError("Empty point cloud")
        print(f"Point cloud size: {len(points)} points")

        # Normalize point cloud to fit in the voxel grid
        # Define the bounding box that contains all points
        min_coord = np.min(points, axis=0) # finds the minimum x, y, and z coordinates
        max_coord = np.max(points, axis=0) # finds the maximum x, y, and z coordinates
        print(f"Point cloud bounds: min={min_coord}, max={max_coord}")

        # Add padding
        extent = max_coord - min_coord
        min_coord = min_coord - pad_factor * extent
        max_coord = max_coord + pad_factor * extent

        # Scale points to voxel grid coordinates
        points_normalized = (points - min_coord) / (max_coord - min_coord) * (self.voxel_resolution - 1)

        # Create empty voxel grid
        voxel_grid = np.zeros((self.voxel_resolution, self.voxel_resolution, self.voxel_resolution), dtype=bool)

        # Fill voxel grid,  a numpy array with shape (N, 3), Each row contains three integers: [x, y, z]
        points_voxel = np.floor(points_normalized).astype(int)

        # Filter points outside the grid (should not happen with proper normalization)
        valid_indices = (
                (points_voxel[:, 0] >= 0) & (points_voxel[:, 0] < self.voxel_resolution) &
                (points_voxel[:, 1] >= 0) & (points_voxel[:, 1] < self.voxel_resolution) &
                (points_voxel[:, 2] >= 0) & (points_voxel[:, 2] < self.voxel_resolution)
        )
        points_voxel = points_voxel[valid_indices]
        print(f"Valid points after normalization: {np.sum(valid_indices)} out of {len(points)}")
        # Set occupied voxels
        for p in points_voxel:
            voxel_grid[p[0], p[1], p[2]] = True

        return voxel_grid

    def save_as_binvox(self, voxel_grid, filepath):
        """Save voxel grid in binvox format"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Create binvox model
        model = binvox_rw.Voxels(
            data=voxel_grid,
            dims=[self.voxel_resolution, self.voxel_resolution, self.voxel_resolution],
            translate=[0.0, 0.0, 0.0],
            scale=1.0,
            axis_order='xyz'
        )

        # Write to file
        with open(filepath, 'wb') as f:
            model.write(f)


def visualize_voxel_grid(dataset_instance_dir):
    """Visualize point cloud and voxel files from a dataset directory"""
    # Find files
    dataset_instance_dir = Path(dataset_instance_dir)
    pointcloud_path = dataset_instance_dir / "pointcloud.ply"
    binvox_files = list(dataset_instance_dir.glob("*.binvox"))
    binvox_path = binvox_files[0] if binvox_files else None

    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    # Check if the point cloud already has colors
    if not pcd.has_colors():
        pcd.paint_uniform_color([0, 0.8, 0])  # Only paint green if no colors exist

    # Print point cloud info
    print(f"Point cloud has {len(np.asarray(pcd.points))} points")

    # Load binvox
    with open(binvox_path, 'rb') as f:
        voxel_model = binvox_rw.read_as_3d_array(f)
        voxel_data = voxel_model.data

        print(f"Binvox dimensions: {voxel_model.dims}")
        print(f"Total possible voxels: {voxel_model.dims[0] * voxel_model.dims[1] * voxel_model.dims[2]}")
        print(f"Occupied voxels: {np.sum(voxel_data)}")
        print(
            f"Occupancy ratio: {np.sum(voxel_data) / (voxel_model.dims[0] * voxel_model.dims[1] * voxel_model.dims[2]):.4f}")


    # Create voxel visualization
    voxel_pcd = o3d.geometry.PointCloud()
    voxel_points = np.array(np.where(voxel_data)).T
    voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points.astype(float))
    voxel_pcd.paint_uniform_color([1, 0, 0])  # Red

    # Visualize
    o3d.visualization.draw_geometries([pcd, voxel_pcd,
                                       o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)])

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def main():
    parser = argparse.ArgumentParser(description='Convert CO3D dataset to voxel representations')
    parser.add_argument('--dataset_root', required=True, help='Root directory of CO3D dataset')
    parser.add_argument('--categories', nargs='+', default=['chair'], help='Categories to process (default: all)')
    parser.add_argument('--voxel_resolution', type=int, default=32, help='Resolution of voxel grid (default: 32)')
    parser.add_argument('--visualize', default=False, type=str2bool, help='Visualize a sample of the output voxels')

    args = parser.parse_args()

    # Create and run the pipeline
    pipeline = CO3D_to_Voxel_Pipeline(
        dataset_root=args.dataset_root,
        categories=args.categories,
        voxel_resolution=args.voxel_resolution
    )

    sample_binvox_path = pipeline.run()

    # Visualize a sample if requested
    if args.visualize and sample_binvox_path is not None:
        visualize_voxel_grid(sample_binvox_path)



if __name__ == "__main__":
    main()
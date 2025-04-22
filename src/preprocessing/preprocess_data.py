"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 6 2025
 * Purpose: Preprocesses CO3D data by copying raw files to a processed directory
            and creating metadata files for sequence organization and tracking.
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def preprocess_sequence(sequence_path, processed_sequence_path):
    """Processes a single sequence by copying raw files and creating metadata"""
    raw_sequence = Path(sequence_path)
    processed_sequence = Path(processed_sequence_path)
    processed_sequence.mkdir(parents=True, exist_ok=True)
    
    # Copy directories with raw data
    subdirs = ["images", "masks", "depths", "depth_masks"]
    for sub in subdirs:
        raw_subdir = raw_sequence / sub
        if raw_subdir.exists():
            processed_subdir = processed_sequence / sub
            processed_subdir.mkdir(parents=True, exist_ok=True)
            for file in raw_subdir.iterdir():
                if file.is_file():
                    shutil.copy2(file, processed_subdir / file.name)
    
    # Create metadata from the processed images
    processed_images_dir = processed_sequence / "images"
    if not processed_images_dir.exists():
        return None
        
    frame_files = sorted(os.listdir(processed_images_dir))
    frames = []
    for img_file in frame_files:
        frame_id = os.path.splitext(img_file)[0]
        pose_path = os.path.join("poses", f"{frame_id}_pose.json")
        full_pose_path = processed_sequence / pose_path
        if not full_pose_path.exists():
            continue
            
        frame_data = {
            "image": os.path.join("images", img_file),
            "pose": pose_path,
            "mask": os.path.join("masks", f"{frame_id}.png"),
            "depth": os.path.join("depths", f"{frame_id}.jpg.geometric.png"),
            "depth_mask": os.path.join("depth_masks", f"{frame_id}.png")
        }
        frames.append(frame_data)
        
    if len(frames) < 2:
        return None  # Skip sequences with insufficient frames
        
    meta = {
        "sequence_name": raw_sequence.name,
        "frames": frames
    }
    meta_path = processed_sequence / "sequence_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    return meta

def preprocess_all_sequences():
    """Process all sequences in the raw data directory"""
    RAW_DATA_ROOT = Path("data/models/chair")
    PROCESSED_ROOT = Path("data/processed/chair")
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    
    instances = [d for d in RAW_DATA_ROOT.iterdir() if d.is_dir()]
    for instance in tqdm(instances, desc="Preprocessing sequences"):
        processed_instance = PROCESSED_ROOT / instance.name
        meta = preprocess_sequence(instance, processed_instance)
        if meta is None:
            print(f"Skipping instance {instance.name} due to insufficient frames.")
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_all_sequences()
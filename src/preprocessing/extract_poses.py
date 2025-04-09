import os
import json
import gzip
from pathlib import Path
from tqdm import tqdm

# Global constants:
ANNOTATION_FILE = "data/models/chair/frame_annotations.jgz"
# Processed output: each instance’s poses will be saved under data/processed/chair/<instance>/poses/
PROCESSED_ROOT = Path("data/processed/chair")

def load_frame_annotations(annotation_path):
    with gzip.open(annotation_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def save_pose(sequence_name, frame_filename, R, T):
    out_dir = PROCESSED_ROOT / sequence_name / "poses"
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_data = {"R": R, "T": T}
    pose_path = out_dir / f"{frame_filename}_pose.json"
    with open(pose_path, "w") as f:
        json.dump(pose_data, f, indent=4)

def extract_and_save_poses():
    print("Loading annotations...")
    annotations = load_frame_annotations(ANNOTATION_FILE)
    print("Extracting poses...")
    for frame in tqdm(annotations):
        sequence_name = frame["sequence_name"]
        # e.g. "chair/62_4316_10649/images/frame000001.jpg"
        image_path = frame["image"]["path"]
        frame_filename = os.path.splitext(os.path.basename(image_path))[0]  # e.g., "frame000001"
        R = frame["viewpoint"]["R"]
        T = frame["viewpoint"]["T"]
        save_pose(sequence_name, frame_filename, R, T)
    print("Pose extraction complete.")

if __name__ == "__main__":
    extract_and_save_poses()

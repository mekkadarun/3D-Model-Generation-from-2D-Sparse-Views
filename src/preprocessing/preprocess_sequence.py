from src.preprocessing.extract_poses import extract_and_save_poses
from src.preprocessing.preprocess_data import preprocess_all_sequences

def main():
    print("Running pose extraction...")
    extract_and_save_poses()
    print("Running data preprocessing...")
    preprocess_all_sequences()
    print("All preprocessing complete. Processed data is available under 'data/processed/chair/'.")

if __name__ == "__main__":
    main()

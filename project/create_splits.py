import torch
import os
from dataset import COCOClipDataset

# --- CONFIGURATION ---
DATA_DIR = "./coco2014"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "images/train2014")
VAL_IMG_DIR = os.path.join(DATA_DIR, "images/val2014")
TRAIN_CACHE = "train_cache_clean.pt"
VAL_CACHE = "val_cache_clean.pt"

# Subset sizes (From your report)
TRAIN_SUBSET_SIZE = 82823
VAL_SUBSET_SIZE = 40485

def save_subset_to_txt(dataset, subset_size, filename):
    print(f"Selecting {subset_size} random images for {filename}...")
    
    # Fix seed for reproducibility
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset))[:subset_size]
    
    with open(filename, "w") as f:
        for idx in indices:
            # We need to temporarily access the path logic from dataset
            item = dataset.data[idx]
            image_id = item['image_id']
            
            # Reconstruct path based on dataset folder
            prefix = "train2014" if "train" in dataset.img_dir else "val2014"
            img_name = f"COCO_{prefix}_{image_id:012d}.jpg"
            full_path = os.path.join(dataset.img_dirc, img_name)
            
            # Write to file
            f.write(full_path + "\n")
            
    print(f"✅ Saved {filename} with {subset_size} paths.")

def main():
    # Load full datasets (metadata only, fast)
    print("Loading datasets...")
    train_ds = COCOClipDataset(TRAIN_IMG_DIR, TRAIN_CACHE)
    val_ds = COCOClipDataset(VAL_IMG_DIR, VAL_CACHE)
    
    # Generate files
    save_subset_to_txt(train_ds, TRAIN_SUBSET_SIZE, "train.txt")
    save_subset_to_txt(val_ds, VAL_SUBSET_SIZE, "test.txt")

if __name__ == "__main__":
    main()
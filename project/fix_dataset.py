import torch
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# Based on your check.py output, your structure involves an 'images' subfolder
BASE_DIR = "./coco2014" 
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "images", "train2014")
VAL_IMG_DIR = os.path.join(BASE_DIR, "images", "val2014")

def clean_cache(cache_file, image_folder, split_prefix):
    """
    Loads a cache file, checks if each image exists on disk, 
    and saves a new cache file containing ONLY the existing images.
    """
    if not os.path.exists(cache_file):
        print(f"Skipping {cache_file} (Not found)")
        return

    print(f"--- Cleaning {cache_file} ---")
    print(f"Checking against images in: {image_folder}")
    
    data = torch.load(cache_file)
    original_count = len(data)
    valid_data = []
    
    # Iterate through all cached captions
    for item in tqdm(data):
        image_id = item['image_id']
        
        # Reconstruct the filename exactly how COCO names them
        # Format: COCO_val2014_000000123456.jpg (Zero-padded to 12 digits)
        filename = f"COCO_{split_prefix}_{image_id:012d}.jpg"
        full_path = os.path.join(image_folder, filename)
        
        # Only keep it if the file physically exists
        if os.path.exists(full_path):
            valid_data.append(item)

    # Save the new "Clean" cache
    new_filename = cache_file.replace(".pt", "_clean.pt")
    torch.save(valid_data, new_filename)
    
    print(f"Done.")
    print(f"Original items: {original_count}")
    print(f"Valid items:    {len(valid_data)}")
    print(f"Removed:        {original_count - len(valid_data)} missing images")
    print(f"Saved to:       {new_filename}\n")

if __name__ == "__main__":
    # Clean Training Cache
    clean_cache("train_cache.pt", TRAIN_IMG_DIR, "train2014")
    
    # Clean Validation Cache
    clean_cache("val_cache.pt", VAL_IMG_DIR, "val2014")
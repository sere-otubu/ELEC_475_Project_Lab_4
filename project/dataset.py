import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import random

# --- Constants & Configuration ---
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SIZE = 224

# --- Helper Function ---
def get_transforms(split="train"):
    """
    Returns transforms. 
    - 'train': Adds augmentation (RandomResizedCrop, Flip, Jitter)
    - 'val':   YOUR ORIGINAL LOGIC (Exact Resize, Deterministic)
    """
    if split == "train":
            return transforms.Compose([
                # REMOVED: RandomResizedCrop
                # REMOVED: RandomHorizontalFlip
                
                # KEEP: Resize (Standard)
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                
                # KEEP: Color Jitter (Makes model robust to lighting, safe for COCO)
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
            ])
    else:
        # Validation stays the same
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])

class COCOClipDataset(Dataset):
    def __init__(self, img_dir, cache_file, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            cache_file (string): Path to the .pt file with cached embeddings.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        # Load the cached text embeddings and metadata
        self.data = torch.load(cache_file) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item['image_id']
        text_embedding = item['text_embedding']
        caption = item['caption_text'] # Kept for verification 

        # Construct filename
        img_filename = f"COCO_train2014_{int(image_id):012d}.jpg" 
        if "val" in self.img_dir:
             img_filename = f"COCO_val2014_{int(image_id):012d}.jpg"
             
        img_path = os.path.join(self.img_dir, img_filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Handle potential missing files gracefully
            print(f"Warning: File not found {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, text_embedding, img_path
    
def verify_dataset(dataset, num_samples=3):
    """
    Displays random image-caption pairs to verify integrity.
    """
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Inverse transform for visualization (undo normalization)
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-m/s for m, s in zip(CLIP_MEAN, CLIP_STD)],
            std=[1/s for s in CLIP_STD]
        ),
        transforms.ToPILImage()
    ])

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        image_tensor, _, caption = dataset[idx]
        
        # Un-normalize for display
        img_disp = inv_normalize(image_tensor)
        
        ax = plt.subplot(1, num_samples, i + 1)
        ax.imshow(img_disp)
        ax.set_title(caption[:30] + "...") # Truncate long captions
        ax.axis('off')
    
    plt.show()

# --- Execution Block ---
if __name__ == "__main__":
    # Only run this if the file is executed directly (not imported)
    
    # UPDATE THIS PATH to match your actual image folder
    # (This should be the folder containing the .jpg files)
    DATA_DIR = "./coco2014" # Example root
    TEST_IMG_DIR = os.path.join(DATA_DIR, "images/train2014")
    CACHE_FILE = "train_cache.pt"

    if os.path.exists(CACHE_FILE) and os.path.exists(TEST_IMG_DIR):
        print("Running dataset verification...")
        dataset = COCOClipDataset(TEST_IMG_DIR, CACHE_FILE, transform=get_transforms())
        verify_dataset(dataset)
    else:
        print("Skipping verification: cache file or image directory not found.")
        print(f"Looking for: {CACHE_FILE} and {TEST_IMG_DIR}")
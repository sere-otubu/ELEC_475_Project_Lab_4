import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import textwrap

# Import your modules
from model import CLIPModel
from dataset import COCOClipDataset, get_transforms

# --- CONFIGURATION ---
BASE_DIR = "./coco2014"
VAL_IMG_DIR = os.path.join(BASE_DIR, "images/val2014")
VAL_CACHE = "val_cache_clean.pt"
# MODEL_PATH = "best_model.pt"
# MODEL_PATH = "best_model_mod1.pt"
MODEL_PATH = "best_model_mod2.pt"

NUM_EXAMPLES = 2
TOP_K = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_caption_safely(dataset, index):
    """
    Robustly extracts a caption from the dataset, handling Subsets and missing keys.
    """
    # 1. Recursively unwrap Subsets to get to the real dataset
    while isinstance(dataset, Subset):
        index = dataset.indices[index]
        dataset = dataset.dataset

    # 2. Try to access the raw data
    if hasattr(dataset, 'data'):
        item = dataset.data[index]
        
        if isinstance(item, dict):
            for key in ['caption', 'text', 'sentence', 'caption_text']:
                if key in item:
                    return item[key]
            # print(f"\nDEBUG ERROR: Item keys found: {list(item.keys())}")
            return "Error: Caption Key Missing"
            
        elif isinstance(item, (list, tuple)):
            for element in item:
                if isinstance(element, str):
                    return element
            return str(item)

    elif hasattr(dataset, 'captions'):
        return dataset.captions[index]

    return "Error: Could not find raw data in dataset"

def visualize_retrieval(model, dataset):
    model.eval()
    
    # 1. Build Index on FULL Dataset
    print(f"Building index of ALL {len(dataset)} validation images...")
    print("(This involves running the model on every image, so please wait...)")
    
    # Use the full dataset here
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_image_embeds = []
    all_image_paths = [] 
    
    with torch.no_grad():
        for batch in loader:
            # Handle return values (image, embed, path)
            if len(batch) == 3:
                images, _, paths = batch 
            else:
                images, _, _ = batch
                paths = ["Unknown"] * len(images)

            images = images.to(device)
            img_out, _ = model(images)
            img_out = img_out / img_out.norm(dim=1, keepdim=True)
            
            all_image_embeds.append(img_out)
            all_image_paths.extend(paths)
            
    all_image_embeds = torch.cat(all_image_embeds)
    print("Index built successfully.")

    # 2. Run Queries
    # Pick random indices from the FULL dataset
    query_indices = random.sample(range(len(dataset)), NUM_EXAMPLES)
    
    print(f"\nVisualizing {NUM_EXAMPLES} queries...")
    
    for idx in query_indices:
        # Access directly from dataset
        item = dataset[idx]
        _, text_embed, true_path = item
        
        # Get caption from full dataset
        caption_text = get_caption_safely(dataset, idx)
        
        text_embed = text_embed.to(device).unsqueeze(0)
        text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)
        
        # Calculate similarity against ALL images
        sims = text_embed @ all_image_embeds.t()
        scores, top_indices = sims.topk(TOP_K, dim=1)
        
        # --- PLOTTING ---
        fig, axes = plt.subplots(1, TOP_K + 1, figsize=(15, 5))
        
        # Add the Caption as the Main Title
        wrapped_caption = "\n".join(textwrap.wrap(f"Query: {caption_text}", width=80))
        fig.suptitle(wrapped_caption, fontsize=14, fontweight='bold', y=0.95)
        
        # Plot 0: Ground Truth
        try:
            true_img = Image.open(true_path).convert("RGB")
            axes[0].imshow(true_img)
            axes[0].set_title("Ground Truth\n(Target)", color='green', fontsize=10)
        except Exception as e:
            axes[0].text(0.5, 0.5, "Img Error", ha='center')
        axes[0].axis("off")
        
        # Plot 1-5: Retrieved
        for i, match_idx in enumerate(top_indices[0]):
            match_path = all_image_paths[match_idx]
            score = scores[0][i].item()
            
            # --- MODIFICATION: Fetch Caption Snippet ---
            # match_idx is the index in the dataset/loader
            retrieved_idx = match_idx.item()
            retrieved_caption = get_caption_safely(dataset, retrieved_idx)
            
            # Create a short snippet (e.g., first 30 chars)
            snippet = textwrap.shorten(retrieved_caption, width=30, placeholder="...")
            
            try:
                img = Image.open(match_path).convert("RGB")
                axes[i+1].imshow(img)
            except:
                continue
            
            is_correct = (match_path == true_path)
            color = 'green' if is_correct else 'red'
            
            # Add snippet to the title
            axes[i+1].set_title(f"Rank {i+1}\nScore: {score:.3f}\n{snippet}", color=color, fontsize=8)
            axes[i+1].axis("off")
            
            if is_correct:
                for spine in axes[i+1].spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(3)

        plt.tight_layout()
        plt.subplots_adjust(top=0.80)
        
        save_name = f"retrieval_example_{idx}.png"
        plt.savefig(save_name)
        plt.show()
        print(f"Saved {save_name}")

def main():
    model = CLIPModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Use split="val" for deterministic transforms (no random crops)
    ds = COCOClipDataset(VAL_IMG_DIR, VAL_CACHE, transform=get_transforms(split="val"), subset_file="subtest_val.txt")
    visualize_retrieval(model, ds)

if __name__ == "__main__":
    main()
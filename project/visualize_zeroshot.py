import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import textwrap
import numpy as np

# Import your modules
from model import CLIPModel
from dataset import COCOClipDataset, get_transforms
from torch.utils.data import Subset

# --- CONFIGURATION ---
BASE_DIR = "./coco2014"
VAL_IMG_DIR = os.path.join(BASE_DIR, "images/val2014")
VAL_CACHE = "val_cache_clean.pt"
# MODEL_PATH = "best_model.pt"
# MODEL_PATH = "best_model_mod1.pt"
MODEL_PATH = "best_model_mod2.pt"

# How many different images to visualize
NUM_EXAMPLES = 1
# How many text options to give the model per image (1 Correct + N Distractors)
NUM_CLASSES = 5 

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
            return "Error: Caption Key Missing"
        elif isinstance(item, (list, tuple)):
            for element in item:
                if isinstance(element, str):
                    return element
            return str(item)
    elif hasattr(dataset, 'captions'):
        return dataset.captions[index]
    return "Error: Could not find raw data"

def visualize_zeroshot_classification(model, dataset):
    model.eval()
    
    # Pick random indices for the images we want to test
    image_indices = random.sample(range(len(dataset)), NUM_EXAMPLES)
    
    print(f"Visualizing {NUM_EXAMPLES} Zero-Shot Classification examples...")

    for idx in image_indices:
        # 1. Get the Query Image and its Ground Truth Text
        item = dataset[idx]
        image_tensor, true_text_embed, img_path = item
        true_caption = get_caption_safely(dataset, idx)
        
        # Prepare Image Batch
        image_tensor = image_tensor.to(device).unsqueeze(0) # [1, C, H, W]
        
        # 2. Get Distractor Texts (Random other captions)
        distractor_indices = random.sample(range(len(dataset)), NUM_CLASSES - 1)
        
        # Ensure we didn't accidentally pick the same index
        while idx in distractor_indices:
            distractor_indices = random.sample(range(len(dataset)), NUM_CLASSES - 1)

        candidate_embeds = [true_text_embed]
        candidate_captions = [true_caption]
        
        for dist_idx in distractor_indices:
            _, dist_embed, _ = dataset[dist_idx]
            candidate_embeds.append(dist_embed)
            candidate_captions.append(get_caption_safely(dataset, dist_idx))
            
        # Stack embeddings: [NUM_CLASSES, Embed_Dim]
        text_stack = torch.stack(candidate_embeds).to(device)
        
        # 3. Run Model
        with torch.no_grad():
            # Get Image Feature
            img_out, _ = model(image_tensor)
            
            # Normalize
            img_out = img_out / img_out.norm(dim=1, keepdim=True)
            text_stack = text_stack / text_stack.norm(dim=1, keepdim=True)
            
            # Compute Similarity (Logits)
            # Shape: [1, Embed_Dim] @ [Embed_Dim, NUM_CLASSES] -> [1, NUM_CLASSES]
            logits = img_out @ text_stack.t()
            
            # Compute Softmax Probabilities
            probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
            
        # 4. Visualization
        # Create a figure with 2 columns: Image on Left, Bar Chart on Right
        fig, (ax_img, ax_chart) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot Image
        try:
            pil_img = Image.open(img_path).convert("RGB")
            ax_img.imshow(pil_img)
        except:
            ax_img.text(0.5, 0.5, "Image Not Found", ha='center')
        ax_img.axis("off")
        ax_img.set_title("Input Image", fontweight='bold')

        # Plot Bar Chart
        y_pos = np.arange(len(candidate_captions))
        
        # Wrap text for the y-axis labels
        wrapped_labels = ["\n".join(textwrap.wrap(c, width=40)) for c in candidate_captions]
        
        # Color the bars: Green for Ground Truth (index 0), Grey for others
        colors = ['green'] + ['grey'] * (NUM_CLASSES - 1)
        
        ax_chart.barh(y_pos, probs, align='center', color=colors)
        ax_chart.set_yticks(y_pos)
        ax_chart.set_yticklabels(wrapped_labels)
        ax_chart.invert_yaxis()  # Labels read top-to-bottom
        ax_chart.set_xlabel('Probability Score')
        ax_chart.set_title(f'Zero-Shot Classification (Image-to-Text) {MODEL_PATH}')
        
        # Add the percentage values at the end of bars
        for i, v in enumerate(probs):
            ax_chart.text(v + 0.01, i, f"{v*100:.1f}%", color='black', va='center')

        plt.tight_layout()
        save_name = f"zeroshot_example_{idx}.png"
        plt.savefig(save_name)
        print(f"Saved {save_name}")

def main():
    print(f"Loading Model from {MODEL_PATH}...")
    model = CLIPModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    print("Preparing Dataset...")
    # Use split="val" to ensure deterministic transforms (center crop)
    ds = COCOClipDataset(VAL_IMG_DIR, VAL_CACHE, transform=get_transforms(split="val"))
    
    visualize_zeroshot_classification(model, ds)

if __name__ == "__main__":
    main()
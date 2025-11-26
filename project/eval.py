import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os

# Import your modules
from model import CLIPModel
from dataset import COCOClipDataset, get_transforms

# --- CONFIGURATION ---
BASE_DIR = "./coco2014"
VAL_IMG_DIR = os.path.join(BASE_DIR, "images/val2014")
VAL_CACHE = "val_cache_clean.pt"
MODEL_PATH = "best_model.pt"
# MODEL_PATH = "best_model_mod1.pt"
# MODEL_PATH = "best_model_mod2.pt"

# LIMIT EVALUATION SIZE TO PREVENT CRASHING
# 1,000 is a standard sample size for quick evaluation
EVAL_SIZE = 1000 
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(model, loader):
    """
    Feeds all images and texts through the model and stores the embeddings.
    """
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    
    with torch.no_grad():
        for images, text_embeddings, _ in tqdm(loader, desc="Generating Embeddings"):
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)
            
            # Get Image Features
            img_out, _ = model(images)
            
            # Normalize them immediately to save memory and prep for Cosine Sim
            img_out = img_out / img_out.norm(dim=1, keepdim=True)
            text_out = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
            
            all_image_embeds.append(img_out.cpu())
            all_text_embeds.append(text_out.cpu())
            
    return torch.cat(all_image_embeds), torch.cat(all_text_embeds)

def calculate_recall(image_embeds, text_embeds, k_values=[1, 5, 10]):
    """
    Calculates Recall@K for Image->Text and Text->Image retrieval.
    """
    # 1. Compute Similarity Matrix (Image x Text)
    # Shape: [N_images, N_texts]
    print("Calculating Similarity Matrix...")
    logits = image_embeds @ text_embeds.t()
    
    n_samples = logits.shape[0]
    
    # 2. Create Ground Truth (The diagonal is the correct match)
    # The 0th image matches the 0th text, 1st matches 1st, etc.
    targets = torch.arange(n_samples)
    
    results = {}
    
    # --- TEXT TO IMAGE RETRIEVAL (Given Text, find Image) ---
    # We look along the columns (images) for each row (text)
    # Transpose logits so rows=texts, cols=images
    logits_t2i = logits.t()
    
    print("\n--- Text-to-Image Retrieval Scores ---")
    for k in k_values:
        # Get the indices of the top K scores
        _, topk_indices = logits_t2i.topk(k, dim=1)
        
        # Check if the correct index is in the top K
        correct = topk_indices.eq(targets.view(-1, 1).expand_as(topk_indices))
        
        # Calculate accuracy
        recall = correct.sum().float() / n_samples
        print(f"R@{k}: {recall.item() * 100:.2f}%")
        results[f"T2I_R@{k}"] = recall.item()

    return results

def main():
    print(f"Loading Model from {MODEL_PATH}...")
    model = CLIPModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    print("Preparing Dataset...")
    full_ds = COCOClipDataset(VAL_IMG_DIR, VAL_CACHE, transform=get_transforms(), subset_file="subtest_val.txt")
    
    # Select a random subset for evaluation
    indices = torch.randperm(len(full_ds))[:EVAL_SIZE]
    subset_ds = Subset(full_ds, indices)
    
    loader = DataLoader(subset_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(subset_ds)} random pairs.")
    
    # 1. Get all embeddings
    img_embs, txt_embs = get_embeddings(model, loader)
    
    # 2. Calculate Metrics
    calculate_recall(img_embs, txt_embs)

if __name__ == "__main__":
    main()
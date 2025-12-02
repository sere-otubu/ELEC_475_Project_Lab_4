#########################################################################################################
#
#   ELEC 475 - Lab 4
#   Erhowvosere Otubu - 20293052
#   Mihran Asadullah - 20285090
#   Fall 2025
#

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# Import your modules
from model import CLIPModel
from dataset import COCOClipDataset, get_transforms

# --- CONFIGURATION ---
BASE_DIR = "./coco2014"
VAL_IMG_DIR = os.path.join(BASE_DIR, "images/val2014")
VAL_CACHE = "val_cache_clean.pt"
# Uncomment the model you wish to evaluate
MODEL_PATH = "best_model.pt"
# MODEL_PATH = "best_model_mod1.pt"
# MODEL_PATH = "best_model_mod2.pt"

BATCH_SIZE = 32
RECALL_BATCH_SIZE = 1000 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(model, loader):
    """
    Feeds all images and texts through the model and stores the embeddings + Paths.
    """
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    all_paths = []
    
    with torch.no_grad():
        # Capture the third return value (paths)
        for batch in tqdm(loader, desc="Generating Embeddings"):
            if len(batch) == 3:
                images, text_embeddings, paths = batch
            else:
                # Fallback if dataset structure varies
                images, text_embeddings = batch[0], batch[1]
                paths = ["unknown"] * len(images)

            images = images.to(device)
            text_embeddings = text_embeddings.to(device)
            
            # Get Image Features
            img_out, _ = model(images)
            
            # Normalize
            img_out = img_out / img_out.norm(dim=1, keepdim=True)
            text_out = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
            
            all_image_embeds.append(img_out.cpu())
            all_text_embeds.append(text_out.cpu())
            all_paths.extend(paths)
            
    return torch.cat(all_image_embeds), torch.cat(all_text_embeds), np.array(all_paths)

def batch_recall_compute(query_embeds, target_embeds, query_paths, target_paths, k_values, prefix=""):
    """
    Computes recall using Path Matching (Handles Duplicates).
    """
    num_queries = query_embeds.shape[0]
    correct_counts = {k: 0 for k in k_values}
    
    # Pre-encode paths to integers for faster comparison
    # We combine both lists to ensure consistent ID mapping
    combined_paths = np.concatenate([query_paths, target_paths])
    _, unique_ids = np.unique(combined_paths, return_inverse=True)
    
    # Split back
    query_ids = torch.from_numpy(unique_ids[:len(query_paths)])
    target_ids = torch.from_numpy(unique_ids[len(query_paths):])
    
    # Try moving targets to GPU
    try:
        target_embeds_gpu = target_embeds.to(device)
        target_ids_gpu = target_ids.to(device)
        use_gpu = True
    except RuntimeError:
        print("Targets too large for GPU, using CPU...")
        target_embeds_gpu = target_embeds
        target_ids_gpu = target_ids
        use_gpu = False

    print(f"\nComputing {prefix} Recall (Batch Size: {RECALL_BATCH_SIZE})...")
    
    for i in tqdm(range(0, num_queries, RECALL_BATCH_SIZE)):
        end = min(i + RECALL_BATCH_SIZE, num_queries)
        
        # Batch Data
        queries_chunk = query_embeds[i:end]
        current_query_ids = query_ids[i:end]
        
        if use_gpu:
            queries_chunk = queries_chunk.to(device)
            current_query_ids = current_query_ids.to(device)
            
        # 1. Similarity Matrix
        sim_matrix = queries_chunk @ target_embeds_gpu.t()
        
        # 2. Top K Indices
        max_k = max(k_values)
        _, topk_indices = sim_matrix.topk(max_k, dim=1)
        
        # 3. Retrieve the IDs of the Top K hits
        # topk_indices is [Batch, K]
        # We gather the Image IDs corresponding to these indices
        topk_ids = target_ids_gpu[topk_indices] # Shape: [Batch, K]
        
        # 4. Check Correctness
        # Does the retrieved ID match the Query ID?
        # current_query_ids.view(-1, 1) -> [Batch, 1]
        correct_mask = topk_ids.eq(current_query_ids.view(-1, 1))
        
        for k in k_values:
            # Check if ANY of the top k matches
            hits = correct_mask[:, :k].any(dim=1).sum().item()
            correct_counts[k] += hits

        del sim_matrix, topk_indices, topk_ids
        if use_gpu:
            del queries_chunk
            torch.cuda.empty_cache()

    print(f"\n{prefix} Results:")
    for k in k_values:
        score = correct_counts[k] / num_queries
        print(f"R@{k}: {score * 100:.2f}%")

def calculate_recall(image_embeds, text_embeds, paths, k_values=[1, 5, 10]):
    # 1. Image-to-Text
    # Query: Image, Target: Text
    # Correct if: Text's associated Image Path == Query Image Path
    batch_recall_compute(image_embeds, text_embeds, paths, paths, k_values, prefix="Image-to-Text")

    # 2. Text-to-Image
    # Query: Text, Target: Image
    # Correct if: Image's Path == Query Text's associated Image Path
    batch_recall_compute(text_embeds, image_embeds, paths, paths, k_values, prefix="Text-to-Image")

def main():
    print(f"Loading Model from {MODEL_PATH}...")
    model = CLIPModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    print("Preparing Dataset...")
    full_ds = COCOClipDataset(VAL_IMG_DIR, VAL_CACHE, transform=get_transforms(), subset_file="subtest_val.txt")
    loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Evaluating on {len(full_ds)} pairs.")
    
    # 1. Get embeddings AND paths
    img_embs, txt_embs, paths = get_embeddings(model, loader)
    
    # 2. Calculate Metrics using paths for ground truth
    calculate_recall(img_embs, txt_embs, paths)

if __name__ == "__main__":
    main()
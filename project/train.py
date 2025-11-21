import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

# Import our custom modules
from model import CLIPModel
from dataset import COCOClipDataset, get_transforms

# --- CONFIGURATION ---
# Use the same base directory you found earlier
DATA_DIR = "./coco2014" # Example root
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "images/train2014")
VAL_IMG_DIR = os.path.join(DATA_DIR, "images/val2014")

# Hyperparameters
BATCH_SIZE = 64        
LEARNING_RATE = 1e-4   
EPOCHS = 5             
TEMPERATURE = 0.07     

# Paths
TRAIN_CACHE = "train_cache.pt"
VAL_CACHE = "val_cache.pt"
SAVE_PATH = "best_model.pt"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def info_nce_loss(image_features, text_features, temperature=0.07):
    """
    Calculates the symmetric InfoNCE (Contrastive) Loss[cite: 40, 73].
    """
    # Normalize features (Cosine Similarity requires normalized vectors)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # Similarity Matrix (logits)
    logits = (image_features @ text_features.t()) / temperature

    # Labels (Diagonal is the positive match)
    labels = torch.arange(len(logits)).to(device)

    # Symmetric Cross Entropy Loss
    loss_i = nn.functional.cross_entropy(logits, labels)      # Image -> Text
    loss_t = nn.functional.cross_entropy(logits.t(), labels)  # Text -> Image
    
    return (loss_i + loss_t) / 2

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    
    loop = tqdm(loader, leave=True, desc="Training")
    for images, text_embeddings, _ in loop:
        images = images.to(device)
        text_embeddings = text_embeddings.to(device)

        optimizer.zero_grad()

        # Forward Pass (Image Encoder only)
        image_out, _ = model(images) 
        
        # Loss vs Cached Text Embeddings
        loss = info_nce_loss(image_out, text_embeddings, TEMPERATURE)

        # Backward Pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"Train Loss: {loss.item():.4f}")

    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_loss = 0
    
    # No gradient calculation needed for validation
    with torch.no_grad():
        for images, text_embeddings, _ in loader:
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)
            
            image_out, _ = model(images)
            loss = info_nce_loss(image_out, text_embeddings, TEMPERATURE)
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main():
    print(f"Using device: {device}")
    print(f"Training on: {TRAIN_IMG_DIR}")
    print(f"Validating on: {VAL_IMG_DIR}")

    # 1. Setup DataLoaders
    # Check if cache files exist
    if not os.path.exists(TRAIN_CACHE) or not os.path.exists(VAL_CACHE):
        print("❌ Error: Cache files not found. Please run preprocess_data.py first.")
        return

    train_ds = COCOClipDataset(TRAIN_IMG_DIR, TRAIN_CACHE, transform=get_transforms())
    val_ds = COCOClipDataset(VAL_IMG_DIR, VAL_CACHE, transform=get_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. Initialize Model
    model = CLIPModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        avg_train_loss = train_one_epoch(model, train_loader, optimizer)
        train_losses.append(avg_train_loss)
        
        # Validate
        avg_val_loss = validate(model, val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model based on Validation Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⬇️ Validation Loss improved. Model saved to {SAVE_PATH}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.2f} minutes.")

    # 4. Plotting 
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('InfoNCE Loss')
    plt.title('CLIP Fine-tuning Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == "__main__":
    main()
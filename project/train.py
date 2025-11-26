import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import your custom modules
from model import CLIPModel
from dataset import COCOClipDataset, get_transforms

# ==========================================
#        MASTER CONFIGURATION
# ==========================================
# Choose your mode: "baseline", "mod_1", or "mod_2"
EXPERIMENT_MODE = "best" 

# Directory Settings
DATA_DIR = "./coco2014" 
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "images/train2014")
VAL_IMG_DIR = os.path.join(DATA_DIR, "images/val2014")
TRAIN_CACHE = "train_cache_clean.pt"
VAL_CACHE = "val_cache_clean.pt"
SAVE_PATH = "best_model.pt"
TRAIN_TXT = "subset_train.txt"
TEST_TXT = "subset_val.txt"
# MODEL_PATH = "best_model_mod1.pt"
# MODEL_PATH = "best_model_mod2.pt"

# Subset Settings (Keep True for consistent reporting)
USE_SUBSET = True       

# Hardware Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
#    AUTO-CONFIGURATION (DO NOT EDIT)
# ==========================================
if EXPERIMENT_MODE == "baseline":
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    USE_ADAMW = False
    USE_SCHEDULER = False
    USE_AUGMENTATION = False
    UNFREEZE_LAYER4 = False
    print(f"MODE: BASELINE (Frozen Backbone, No Augmentation)")

elif EXPERIMENT_MODE == "mod_1":
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    USE_ADAMW = False # Typically baseline optimizer is kept for simple aug tests
    USE_SCHEDULER = False
    USE_AUGMENTATION = True
    UNFREEZE_LAYER4 = False
    print(f"MODE: Modification 1 (Frozen Backbone, Weight Decay, Data Augmentation)")

elif EXPERIMENT_MODE == "mod_2":
    BATCH_SIZE = 64  # Increased for RTX 2000 Ada
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    USE_ADAMW = True
    USE_SCHEDULER = True
    USE_AUGMENTATION = True
    UNFREEZE_LAYER4 = True
    print(f" MODE: Modification 2 (Unfrozen Layer 4, Weight Decay, AdamW, Scheduler, Augmentation)")

# ==========================================
#              TRAINING CODE
# ==========================================

def info_nce_loss(image_features, text_features, temperature=0.07):
    # Normalize features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # Similarity Matrix & Labels
    logits = (image_features @ text_features.t()) / temperature
    labels = torch.arange(len(logits)).to(DEVICE)

    # Symmetric Loss
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    loop = tqdm(loader, leave=True, desc="Training")
    
    for images, text_embeddings, _ in loop:
        images = images.to(DEVICE)
        text_embeddings = text_embeddings.to(DEVICE)

        optimizer.zero_grad()
        image_out, _ = model(images) 
        loss = info_nce_loss(image_out, text_embeddings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"Train Loss: {loss.item():.4f}")

    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, text_embeddings, _ in loader:
            images = images.to(DEVICE)
            text_embeddings = text_embeddings.to(DEVICE)
            image_out, _ = model(images)
            loss = info_nce_loss(image_out, text_embeddings)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {DEVICE}")

    # 1. SETUP DATASETS
    # Transform logic: If augmentation is ON, use 'train' split logic, else use 'val' logic
    print("Loading datasets from split files...")

    train_transform_split = "train" if USE_AUGMENTATION else "val"
    
    train_ds = COCOClipDataset(TRAIN_IMG_DIR, TRAIN_CACHE, transform=get_transforms(split=train_transform_split), subset_file=TRAIN_TXT)
    val_ds = COCOClipDataset(VAL_IMG_DIR, VAL_CACHE, transform=get_transforms(split="val"), subset_file=TEST_TXT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. INITIALIZE MODEL
    model = CLIPModel().to(DEVICE)

    # Handle Freezing/Unfreezing Override
    # Note: model.py defaults to Unfreezing Layer 4. 
    # If we are in 'baseline' or 'augment' mode, we must RE-FREEZE it.
    if not UNFREEZE_LAYER4:
        print("Locking Backbone...")
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    
    # 3. SETUP OPTIMIZER & SCHEDULER
    if USE_ADAMW:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    else:
        # Baseline uses standard Adam
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = None
    if USE_SCHEDULER:
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. TRAINING LOOP
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting {EXPERIMENT_MODE.upper()} training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        avg_train_loss = train_one_epoch(model, train_loader, optimizer)
        avg_val_loss = validate(model, val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Scheduler Step
        current_lr = LEARNING_RATE
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⬇️ Model saved to {SAVE_PATH}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    plt.title(f'Training Results ({EXPERIMENT_MODE.upper()})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == "__main__":
    main()
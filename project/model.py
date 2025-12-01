import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPTextModel

class ProjectionHead(nn.Module):
    """
    Project features to a lower dimensional embedding space.
    Specification: Two linear layers with GELU activation.
    Input: 2048 (ResNet50 feature dim) -> Output: 512 (CLIP embedding dim)
    """
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)

class CLIPModel(nn.Module):
    def __init__(self, freeze_text_encoder=True, train_only_layer4=False): 
        super().__init__()
        
        # 1. Image Encoder (ResNet50)
        # We load the weights from ImageNet
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove the classification head (fc layer)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # --- TRAINING CONFIGURATION ---
        if train_only_layer4:
            # MODE: Modification 2 (Partial Freezing)
            # 1. Freeze EVERYTHING first
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            
            # 2. Unfreeze ONLY Layer 4
            for param in self.image_encoder[7].parameters():
                param.requires_grad = True
        else:
            # MODE: Baseline & Mod 1 (Full Training)
            # Ensure everything is trainable (Standard PyTorch behavior)
            for param in self.image_encoder.parameters():
                param.requires_grad = True
        
        # 2. Projection Head (Trainable)
        self.image_projection = ProjectionHead(input_dim=2048, output_dim=512)

        # 3. Text Encoder (Frozen)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, images, input_ids=None, attention_mask=None):
        """
        Args:
            images: Tensor of shape (batch_size, 3, 224, 224)
            input_ids: (Optional) Tokenized text IDs
            attention_mask: (Optional) Attention mask for text
        Returns:
            image_embeddings: (batch_size, 512)
            text_embeddings: (batch_size, 512) (if input_ids provided, else None)
        """
        # 1. Encode Images
        # ResNet output shape: (batch, 2048, 1, 1) -> Flatten to (batch, 2048)
        features = self.image_encoder(images).flatten(1)
        image_embeddings = self.image_projection(features)

        # 2. Encode Text (if provided)
        text_embeddings = None
        if input_ids is not None:
            # We use the pooler_output from the transformer
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeddings = text_out.pooler_output

        return image_embeddings, text_embeddings

if __name__ == "__main__":
    # Sanity Check
    print("Initializing CLIP Model...")
    model = CLIPModel()
    
    # Create dummy input
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    img_emb, _ = model(dummy_image)
    
    print("\n--- Model Verification ---")
    print(f"Image Encoder Output Shape: {img_emb.shape} (Expected: [1, 512])")
    
    # Verify freezing
    text_params_grad = any(p.requires_grad for p in model.text_encoder.parameters())
    img_params_grad = any(p.requires_grad for p in model.image_encoder.parameters())
    
    print(f"Text Encoder Frozen: {not text_params_grad} (Expected: True)")
    print(f"Image Encoder Trainable: {img_params_grad} (Expected: True)")
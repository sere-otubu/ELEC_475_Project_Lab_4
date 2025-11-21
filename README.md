# ELEC 475 Lab 4: Fine-tuning CLIP

## 1. Introduction
[cite_start]This project implements a Contrastive Language-Image Pre-training (CLIP) model by fine-tuning a **ResNet50** image encoder to align with a frozen pretrained Transformer text encoder[cite: 34, 35, 36]. [cite_start]The model is trained on the **MS COCO 2014** dataset using the InfoNCE loss function to learn a joint embedding space for images and captions[cite: 37, 40].

[cite_start]**Authors:** Ali Zavareh, Justin Jacob, and Michael Greenspan[cite: 16, 17].
[cite_start]**Date:** November 2025[cite: 18].

## 2. Project Structure
```text
project_folder/
│
├── coco2014/              # Dataset directory (auto-created by download script)
│   ├── train2014/         # Training images [cite: 46]
│   ├── val2014/           # Validation images [cite: 47]
│   └── annotations/       # Caption JSON files [cite: 48]
│
├── download_data.py       # Script to download COCO 2014 via KaggleHub
├── preprocess_data.py     # Caches text embeddings to .pt files 
├── dataset.py             # Custom PyTorch Dataset class & verification
├── model.py               # CLIP model definition (ResNet50 + Projection)
├── train.py               # Main training loop
├── requirements.txt       # Python dependencies
├── Train.txt              # Command used to start training [cite: 108]
├── Test.txt               # Command used to start evaluation [cite: 109]
└── README.md              # Project documentation
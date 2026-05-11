# Crowd Counting with Density Map Regression

## AI Powered Crowd Estimation from Images

---

A deep learning model for accurate crowd counting that predicts density maps from crowd images. Built using VGG16-BN encoder with dilated convolutional decoder, trained on ShanghaiTech Part A and Part B datasets. The model outputs both a density map visualization and the total count of people in the image.

---

## Overview

This project implements crowd counting using density map regression. The model takes a crowd image as input and generates a density map where each pixel represents crowd density. Summing all pixel values gives the total number of people. The architecture uses a pretrained VGG16-BN backbone for feature extraction and a custom decoder with dilated convolutions and upsampling layers for high-quality density map estimation.

---

## Dataset

| Dataset | Images | Average Count | Purpose |
|---------|--------|---------------|---------|
| ShanghaiTech Part A | 482 | 501 | Dense crowd scenes |
| ShanghaiTech Part B | 716 | 123 | Sparse street crowds |

---

## Model Architecture

| Component | Model | Parameters | Task |
|-----------|-------|------------|------|
| Encoder | VGG16-BN (ImageNet pretrained) | 14.7M | Feature extraction |
| Decoder | Dilated Convolutions + Upsampling | 3.5M | Density map generation |
| Total | End-to-end | 9.2M | Crowd counting |

---

## Results

| Dataset | MAE | MSE | Error Percentage |
|--------|-----|-----|------------------|
| Part A Test Set | 24.00 | 1202.17 | 5.3% |
| Part B Test Set | 18.90 | 400.23 | 15.4% |
| Cross-Dataset | 20.97 | 1267.77 | - |

---

## Features

- End to end density map regression for crowd counting
- Pretrained VGG16-BN backbone for robust feature extraction
- Dilated convolutions maintain spatial resolution
- Custom loss function combining MSE and Absolute Count Error
- Trained on ShanghaiTech Part A (300 images) and Part B (400 images)
- Real time inference on 384x512 resolution images
- Generates density map visualization alongside count
- Batch-1 training optimized for Google Colab Tesla T4 GPU
- Checkpoint saving every 20 epochs with early stopping

---

## Project Files

| File | Purpose |
|------|---------|
| `crowd_counting_dmcount.ipynb` | Complete training and evaluation code |
| `crowd_counter.pth` | Trained model weights |

---

## How It Works

1. Input crowd image is loaded and resized to 384x512 pixels
2. VGG16-BN encoder extracts multi-scale features from the image
3. Decoder with dilated convolutions processes features
4. Three upsampling layers generate final density map
5. Sum of all pixel values in density map equals total crowd count
6. Density map overlay shows crowd distribution on the image

---

## Setup Instructions

### Step 1: Download Model File

Download the trained model file (```crowd_counter.pth```) from Google Drive:

**Download Link:** https://drive.google.com/file/d/1eNbohPZPNYxt2j5dZOIIftFr7cpHEPTO/view?usp=sharing

### Step 2: Upload Model to Google Colab

Upload the downloaded `crowd_counter.pth` file to your Google Colab environment or any local Python environment.

### Step 3: Install Dependencies

```pip install torch torchvision numpy opencv-python matplotlib pillow```

### Step 4: Load Model and Run Inference

```
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define model architecture
class Counter(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16_bn(weights=None)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:33])
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 1)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Counter().to(device)
model.load_state_dict(torch.load("crowd_counter.pth", map_location=device))
model.eval()

# Predict on an image
image_path = "your_image.jpg"
img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (512, 384))
img_tensor = torch.tensor(img_resized.transpose(2,0,1).astype(np.float32)/255.0).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(img_tensor)
    count = pred.sum().item()

print(f"Predicted crowd count: {count:.0f} people")

# Visualize
plt.imshow(img_resized)
plt.imshow(pred.cpu().squeeze().numpy(), cmap="jet", alpha=0.5)
plt.title(f"Count: {count:.0f}")
plt.show()
```

### Step 5: Use Your Own Images
Replace "your_image.jpg" with the path to any crowd image. The model works best with images containing dense crowds (50+ people).

---

## Limitations

- Trained on 384x512 resolution due to GPU memory constraints  
- ShanghaiTech dataset specific, may not fully generalize to all real-world scenarios  
- Trained on single Tesla T4 GPU with batch size 1  
- Web images with different lighting, angles, or distributions may produce errors  
- Sparse crowds (under 10 people) may be undercounted or overcounted  
- Limited to counting people only, not objects or vehicles  
- No individual person detection or localization  

---

## Future Improvements

- Train on full resolution images with more powerful GPU  
- Implement multi-scale testing for better accuracy across crowd densities  
- Add individual head localization alongside counting  
- Train on multiple datasets for better generalization  
- Deploy as Gradio web application with user interface  
- Add video crowd counting with temporal tracking  
- Experiment with transformer backbones (Swin, ViT)  
- Add object counting capability for vehicles and other objects  

---

## Citation

- Dataset: Zhang et al., *"Single-Image Crowd Counting via Multi-Column Convolutional Neural Network"*, CVPR 2016  

- Method: Wang et al., *"Distribution Matching for Crowd Counting"*, NeurIPS 2020  



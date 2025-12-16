# Image Clustering with Deep Learning Embeddings

## Assignment (h)

**Notebook:** `h_image_clustering_imagebind.ipynb`  
**Author:** Nitish  
**Date:** December 2024

---

## Overview

This notebook demonstrates **Image Clustering** using deep learning embeddings extracted from pretrained CNN models. It uses ResNet50 pretrained on ImageNet to generate semantic image representations for clustering.

---

## Table of Contents

1. Introduction
2. Setup and Installation
3. Load Image Dataset
4. Extract Image Embeddings
5. Clustering Analysis
6. Visualization
7. Evaluation Metrics

---

## Embedding Model

### ResNet50 (Pretrained on ImageNet):
- **Architecture:** 50-layer Residual Network
- **Embedding Dimension:** 2048
- **Pretrained:** ImageNet (1000 classes)
- **Layer Used:** Penultimate layer (before classification)

### Why Deep Embeddings?
- Capture high-level semantic features
- Transfer learning from large-scale training
- Better than raw pixels or handcrafted features

---

## Features Implemented

- Image embedding extraction with PyTorch
- PCA for dimensionality reduction
- UMAP for 2D visualization
- K-Means and Hierarchical clustering
- Cluster composition analysis
- Sample image visualization per cluster

---

## Dataset

### CIFAR-10:
- **Total Images:** 10,000 (test set)
- **Sampled:** 1,000 images
- **Image Size:** 32x32 (resized to 224x224)
- **Classes:** 10

| Class | Category |
|-------|----------|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

---

## Workflow

```
Images → Resize (224x224) → Normalize → ResNet50 → Embeddings (2048D)
                                                          ↓
                                                    PCA (50D)
                                                          ↓
                                                    UMAP (2D)
                                                          ↓
                                                    K-Means
```

---

## Image Preprocessing

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

---

## Quality Metrics

### Internal Metrics:
- **Silhouette Score**
- **Calinski-Harabasz Index**
- **Davies-Bouldin Index**

### External Metrics:
- **Adjusted Rand Index (ARI)**
- **Normalized Mutual Information (NMI)**

---

## Key Visualizations

1. **Sample Images:** One per class
2. **UMAP Projection:** 2D embedding space
3. **Elbow Plot:** Optimal K selection
4. **Cluster Comparison:** True vs predicted
5. **Confusion Matrix:** Class vs cluster
6. **Cluster Samples:** Representative images

---

## Requirements

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn umap-learn Pillow
```

---

## How to Run

1. Open `h_image_clustering_imagebind.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. CIFAR-10 is downloaded automatically

---

## Key Findings

- Deep embeddings capture semantic image features
- Similar objects (vehicles, animals) cluster together
- UMAP reveals clear cluster structure
- K-Means achieves reasonable alignment with true classes

---

## Semantic Groupings Observed

| Cluster Type | Classes |
|--------------|---------|
| Vehicles | Airplane, Automobile, Ship, Truck |
| Animals | Bird, Cat, Deer, Dog, Frog, Horse |

---

## Applications

- Image organization and retrieval
- Visual similarity search
- Content-based image classification
- Dataset exploration and curation
- Duplicate image detection
- Photo album organization

---

## Alternative Embedding Models

| Model | Embedding Dim | Pretrained |
|-------|---------------|------------|
| ResNet50 | 2048 | ImageNet |
| VGG16 | 4096 | ImageNet |
| EfficientNet | 1280 | ImageNet |
| CLIP | 512 | Web images + text |
| ImageBind | 1024 | Multimodal |

---

## Code Example

```python
import torch
from torchvision import models

# Load pretrained model
resnet = models.resnet50(pretrained=True)
embedding_model = torch.nn.Sequential(*list(resnet.children())[:-1])

# Extract embeddings
with torch.no_grad():
    embeddings = embedding_model(images).squeeze()
```

---

## References

- He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
- Krizhevsky, A. (2009). "Learning Multiple Layers of Features from Tiny Images" (CIFAR-10)
- PyTorch: https://pytorch.org/

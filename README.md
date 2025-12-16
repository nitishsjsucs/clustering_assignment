# Clustering Algorithms Assignment

## Overview
This repository contains 9 comprehensive Jupyter notebooks implementing various clustering algorithms and techniques.

## Notebooks

### a) K-Means Clustering from Scratch
**File:** `a_kmeans_from_scratch.ipynb`
- Complete K-Means implementation from scratch
- K-Means++ initialization
- Tested on synthetic and Iris datasets
- Comparison with scikit-learn
- Elbow method and silhouette analysis

### b) Hierarchical Clustering
**File:** `b_hierarchical_clustering.ipynb`
- Agglomerative clustering using scikit-learn
- Dendrogram visualization
- Comparison of linkage methods (single, complete, average, ward)
- Customer segmentation use case

### c) Gaussian Mixture Models (GMM)
**File:** `c_gaussian_mixture_models.ipynb`
- GMM clustering with different covariance types
- Soft clustering with probability assignments
- BIC/AIC model selection
- Comparison with K-Means
- Anomaly detection application

### d) DBSCAN with PyCaret
**File:** `d_dbscan_pycaret.ipynb`
- DBSCAN implementation using PyCaret
- Parameter tuning (eps, min_samples)
- K-distance graph for optimal eps
- Comparison with other clustering algorithms
- Works well on non-convex clusters

### e) Anomaly Detection with PyOD
**File:** `e_anomaly_detection_pyod.ipynb`
- Multiple anomaly detection algorithms (IForest, LOF, KNN, COPOD, etc.)
- Univariate and multivariate anomaly detection
- Time series anomaly detection
- Credit card fraud detection use case
- ROC-AUC and precision-recall evaluation

### f) Time Series Clustering
**File:** `f_timeseries_clustering.ipynb`
- DTW (Dynamic Time Warping) based clustering
- Soft-DTW K-Means
- UCR dataset analysis
- Stock market clustering
- Feature-based time series clustering

### g) Document Clustering with LLM Embeddings
**File:** `g_document_clustering_llm.ipynb`
- Sentence Transformers for document embeddings
- UMAP visualization
- K-Means and hierarchical clustering
- Topic analysis and cluster interpretation
- 5-category document classification

### h) Image Clustering with Deep Embeddings
**File:** `h_image_clustering_imagebind.ipynb`
- ResNet50 for image feature extraction
- CIFAR-10 dataset clustering
- PCA and UMAP dimensionality reduction
- Visual cluster analysis
- Confusion matrix evaluation

### i) Audio Clustering with Deep Embeddings
**File:** `i_audio_clustering_embeddings.ipynb`
- Librosa for audio feature extraction
- MFCC and spectral features
- Synthetic audio dataset (sine, square, chirp, noise)
- Spectrogram visualization
- K-Means and hierarchical clustering

## Clustering Quality Metrics Used

### Internal Metrics (No ground truth needed)
- **Silhouette Score:** Measures cluster separation (-1 to 1, higher is better)
- **Calinski-Harabasz Index:** Ratio of between/within cluster dispersion (higher is better)
- **Davies-Bouldin Index:** Average cluster similarity (lower is better)

### External Metrics (Ground truth required)
- **Adjusted Rand Index (ARI):** Similarity between predicted and true labels (-1 to 1)
- **Normalized Mutual Information (NMI):** Mutual information normalized by entropy (0 to 1)

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install pycaret pyod tslearn
pip install sentence-transformers umap-learn
pip install torch torchvision librosa
pip install yfinance plotly
```

## How to Run

1. Open each notebook in Google Colab or Jupyter
2. Run all cells sequentially
3. Each notebook is self-contained with its own data generation/loading

## Author
Nitish

## Date
December 2024

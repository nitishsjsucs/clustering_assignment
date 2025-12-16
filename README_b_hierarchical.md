# Hierarchical Clustering

## Assignment (b)

**Notebook:** `b_hierarchical_clustering.ipynb`  
**Author:** Nitish  
**Date:** December 2024

---

## Overview

This notebook demonstrates **Hierarchical Clustering** using scikit-learn and scipy. It explores agglomerative (bottom-up) clustering with different linkage methods and provides comprehensive dendrogram analysis.

---

## Table of Contents

1. Introduction to Hierarchical Clustering
2. Theory and Concepts
3. Dataset Preparation
4. Agglomerative Clustering
5. Dendrogram Analysis
6. Different Linkage Methods
7. Clustering Quality Metrics
8. Real-World Application: Customer Segmentation

---

## Algorithm Description

### Types of Hierarchical Clustering:
- **Agglomerative (Bottom-up):** Start with each point as a cluster, merge closest pairs
- **Divisive (Top-down):** Start with all points in one cluster, recursively split

### Linkage Methods:

| Method | Description | Best For |
|--------|-------------|----------|
| **Single** | Minimum distance between clusters | Elongated clusters |
| **Complete** | Maximum distance between clusters | Compact clusters |
| **Average** | Average distance between all pairs | Balanced approach |
| **Ward** | Minimizes variance within clusters | Spherical clusters |

---

## Features Implemented

- Agglomerative clustering with scikit-learn
- Dendrogram visualization with scipy
- Cophenetic correlation coefficient calculation
- Multiple linkage method comparison
- Optimal cluster number selection

---

## Datasets Used

| Dataset | Samples | Features | Clusters |
|---------|---------|----------|----------|
| Synthetic Blobs | 300 | 2 | 4 |
| Moon-shaped | 300 | 2 | 2 |
| Iris | 150 | 4 | 3 |
| Wine | 178 | 13 | 3 |
| Customer Data | 500 | 4 | 4 |

---

## Quality Metrics

### Internal Metrics:
- **Silhouette Score:** Cluster separation quality
- **Calinski-Harabasz Index:** Between/within cluster ratio
- **Davies-Bouldin Index:** Cluster compactness
- **Cophenetic Correlation:** Dendrogram quality measure

### External Metrics:
- **Adjusted Rand Index (ARI)**
- **Normalized Mutual Information (NMI)**

---

## Key Visualizations

1. **Dendrograms:** Tree-like cluster hierarchy visualization
2. **Linkage Comparison:** Side-by-side comparison of methods
3. **Customer Segmentation:** Real-world application plots
4. **Heatmaps:** Cluster characteristics visualization

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy plotly
```

---

## How to Run

1. Open `b_hierarchical_clustering.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. The notebook generates synthetic data automatically

---

## Key Findings

- **Ward linkage** works best for compact, spherical clusters
- **Single linkage** excels at finding elongated or non-convex clusters
- Dendrograms provide interpretable cluster hierarchy
- No need to specify number of clusters beforehand

---

## Advantages

- No need to specify K beforehand
- Dendrogram provides interpretable visualization
- Can capture hierarchical relationships
- Deterministic results

---

## Limitations

- Computationally expensive O(n²) for large datasets
- Cannot undo merges (agglomerative)
- Sensitive to noise and outliers
- Memory intensive for large datasets

---

## Use Case: Customer Segmentation

The notebook includes a customer segmentation example with:
- Age, Income, Spending Score, Purchase Frequency features
- 4 distinct customer segments identified
- Cluster profiles and business insights

---

## References

- Müllner, D. (2011). "Modern hierarchical, agglomerative clustering algorithms"
- Ward, J. H. (1963). "Hierarchical Grouping to Optimize an Objective Function"

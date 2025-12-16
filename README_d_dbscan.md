# DBSCAN Clustering with PyCaret

## Assignment (d)

**Notebook:** `d_dbscan_pycaret.ipynb`  
**Author:** Nitish  
**Date:** December 2024

---

## Overview

This notebook demonstrates **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** using the **PyCaret** library. DBSCAN is a density-based clustering algorithm that can discover clusters of arbitrary shape and automatically detect outliers.

---

## Table of Contents

1. Introduction to DBSCAN
2. DBSCAN Theory
3. PyCaret Setup
4. Data Preparation
5. DBSCAN with PyCaret
6. Comparing Multiple Clustering Models
7. Parameter Tuning
8. Clustering Quality Metrics
9. Visualization

---

## Algorithm Description

### Key Concepts:

| Term | Description |
|------|-------------|
| **Core Points** | Points with at least `min_samples` neighbors within `eps` radius |
| **Border Points** | Points within `eps` of a core point but with fewer neighbors |
| **Noise Points** | Points that are neither core nor border (labeled as -1) |

### Parameters:
- **eps (ε):** Maximum distance between two points to be considered neighbors
- **min_samples:** Minimum number of points to form a dense region

### Algorithm Steps:
1. For each point, find all neighbors within eps
2. If point has ≥ min_samples neighbors, it's a core point
3. Create clusters by connecting core points that are neighbors
4. Assign border points to nearest core point's cluster
5. Mark remaining points as noise (-1)

---

## Features Implemented

- DBSCAN using PyCaret's clustering module
- K-distance graph for optimal eps selection
- Grid search for parameter tuning
- Comparison with K-Means, Hierarchical, Mean Shift
- Automatic noise/outlier detection

---

## Datasets Used

| Dataset | Samples | Features | True Clusters |
|---------|---------|----------|---------------|
| Moon-shaped | 500 | 2 | 2 |
| Concentric Circles | 500 | 2 | 2 |
| Noisy Blobs | 450 | 2 | 3 + noise |
| Iris | 150 | 4 | 3 |

---

## Parameter Tuning Methods

### 1. K-Distance Graph:
- Plot sorted distances to k-th nearest neighbor
- Elbow point suggests optimal eps value

### 2. Grid Search:
- Test combinations of eps and min_samples
- Evaluate using silhouette score and ARI

---

## Quality Metrics

### Internal Metrics:
- **Silhouette Score** (excluding noise points)
- **Calinski-Harabasz Index**
- **Davies-Bouldin Index**

### External Metrics:
- **Adjusted Rand Index (ARI)**
- **Normalized Mutual Information (NMI)**

### DBSCAN-Specific:
- Number of clusters discovered
- Number of noise points detected

---

## Key Visualizations

1. **K-Distance Graph:** For optimal eps selection
2. **Parameter Sensitivity Heatmaps:** eps vs min_samples
3. **Cluster Visualizations:** With noise points highlighted
4. **Model Comparison:** DBSCAN vs other algorithms

---

## PyCaret Functions Used

```python
from pycaret.clustering import *

setup(data, normalize=True)      # Initialize
create_model('dbscan')           # Create DBSCAN model
assign_model(model)              # Get cluster labels
plot_model(model, plot='cluster') # Visualize
```

---

## Requirements

```bash
pip install pycaret[full] numpy pandas matplotlib seaborn scikit-learn plotly
```

---

## How to Run

1. Open `d_dbscan_pycaret.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. The notebook generates synthetic data automatically

---

## Key Findings

- DBSCAN excels at non-convex clusters (moons, circles)
- Automatic outlier detection is a major advantage
- Parameter selection is critical for good results
- Single linkage hierarchical clustering has similar behavior

---

## Advantages

- No need to specify number of clusters
- Can find arbitrarily shaped clusters
- Robust to outliers (detects them automatically)
- Works well with spatial data

---

## Limitations

- Sensitive to eps and min_samples parameters
- Struggles with varying density clusters
- Not suitable for high-dimensional data
- May merge close clusters

---

## When to Use DBSCAN

✅ Clusters have arbitrary shapes  
✅ Data contains noise/outliers  
✅ Number of clusters is unknown  
✅ Clusters have similar density  

❌ Clusters have varying densities  
❌ High-dimensional data  
❌ Need soft cluster assignments  

---

## References

- Ester, M., et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise"
- PyCaret Documentation: https://pycaret.org/

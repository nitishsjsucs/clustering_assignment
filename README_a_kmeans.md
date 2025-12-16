# K-Means Clustering from Scratch

## Assignment (a)

**Notebook:** `a_kmeans_from_scratch.ipynb`  
**Author:** Nitish  
**Date:** December 2024

---

## Overview

This notebook implements the K-Means clustering algorithm **completely from scratch** without using scikit-learn's KMeans implementation. It demonstrates the core concepts of centroid-based clustering and compares the custom implementation with the standard library.

---

## Table of Contents

1. Introduction to K-Means
2. Theory and Algorithm
3. Implementation from Scratch
4. Testing on Synthetic Data
5. Testing on Real Dataset (Iris)
6. Comparison with Scikit-learn
7. Clustering Quality Metrics
8. Visualizations

---

## Algorithm Description

### K-Means Steps:
1. **Initialization:** Randomly select k data points as initial centroids
2. **Assignment:** Assign each data point to the nearest centroid
3. **Update:** Recalculate centroids as the mean of all points in each cluster
4. **Repeat:** Steps 2-3 until convergence or max iterations reached

### Mathematical Formulation:

**Objective Function (Inertia):**
```
J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
```

Where:
- k = number of clusters
- Cᵢ = set of points in cluster i
- μᵢ = centroid of cluster i

---

## Features Implemented

- **KMeansFromScratch Class** with methods:
  - `fit()` - Train the model
  - `predict()` - Predict cluster labels
  - `fit_predict()` - Train and predict in one step
  - `get_cluster_centers()` - Return centroids

- **Initialization Methods:**
  - Random initialization
  - K-Means++ initialization (smarter centroid selection)

- **Convergence Tracking:**
  - Iteration history
  - Inertia tracking

---

## Datasets Used

| Dataset | Samples | Features | Clusters |
|---------|---------|----------|----------|
| Synthetic Blobs | 500 | 2 | 4 |
| Iris | 150 | 4 | 3 |

---

## Quality Metrics

### Internal Metrics (No ground truth needed):
- **Silhouette Score:** Measures cluster separation (-1 to 1)
- **Calinski-Harabasz Index:** Ratio of between/within cluster dispersion
- **Davies-Bouldin Index:** Average cluster similarity (lower is better)

### External Metrics (Ground truth required):
- **Adjusted Rand Index (ARI):** Agreement with true labels
- **Normalized Mutual Information (NMI):** Mutual information score

---

## Key Visualizations

1. **Clustering Process Animation:** Step-by-step centroid movement
2. **Elbow Method:** Finding optimal K
3. **Silhouette Analysis:** Cluster quality assessment
4. **Comparison Plots:** Our implementation vs scikit-learn

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn plotly
```

---

## How to Run

1. Open `a_kmeans_from_scratch.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. The notebook generates synthetic data automatically

---

## Key Findings

- Custom implementation produces results comparable to scikit-learn
- K-Means++ initialization leads to better convergence
- The algorithm converges in 5-10 iterations typically
- Works well for spherical, well-separated clusters

---

## Limitations of K-Means

- Assumes spherical clusters of similar size
- Sensitive to initial centroid placement
- Requires specifying K in advance
- Sensitive to outliers

---

## References

- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding"

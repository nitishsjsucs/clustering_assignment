# Gaussian Mixture Models (GMM) Clustering

## Assignment (c)

**Notebook:** `c_gaussian_mixture_models.ipynb`  
**Author:** Nitish  
**Date:** December 2024

---

## Overview

This notebook demonstrates **Gaussian Mixture Models (GMM)** for clustering using scikit-learn. GMM is a probabilistic model that provides soft clustering assignments and can model elliptical cluster shapes.

---

## Table of Contents

1. Introduction to GMM
2. Theory and Mathematics
3. Dataset Preparation
4. GMM Implementation
5. Model Selection (BIC/AIC)
6. Comparison with K-Means
7. Clustering Quality Metrics
8. Advanced Applications

---

## Algorithm Description

### Gaussian Mixture Model:

A GMM represents the probability distribution of data as:

```
p(x) = Σₖ πₖ N(x | μₖ, Σₖ)
```

Where:
- K = number of components (clusters)
- πₖ = mixing coefficient (weight) for component k
- μₖ = mean of component k
- Σₖ = covariance matrix of component k

### Expectation-Maximization (EM) Algorithm:

1. **E-step:** Compute responsibilities (soft assignments)
2. **M-step:** Update parameters (means, covariances, weights)

---

## Covariance Types

| Type | Description | Flexibility |
|------|-------------|-------------|
| **Full** | Each component has its own general covariance | Most flexible |
| **Tied** | All components share the same covariance | Moderate |
| **Diagonal** | Each component has diagonal covariance | Axis-aligned |
| **Spherical** | Each component has single variance | Circular clusters |

---

## Features Implemented

- GMM with different covariance types
- Soft clustering with probability assignments
- BIC/AIC model selection
- Uncertainty quantification
- Anomaly detection using GMM
- Sample generation from learned model

---

## Datasets Used

| Dataset | Samples | Features | Clusters |
|---------|---------|----------|----------|
| Spherical Blobs | 500 | 2 | 4 |
| Elliptical Clusters | 500 | 2 | 3 |
| Iris | 150 | 4 | 3 |
| Wine | 178 | 13 | 3 |

---

## Model Selection

### BIC (Bayesian Information Criterion):
- Penalizes model complexity more heavily
- Preferred for selecting number of components

### AIC (Akaike Information Criterion):
- Less penalty for complexity
- May select more components

---

## Quality Metrics

### Model Metrics:
- **Log-likelihood:** Model fit quality
- **BIC/AIC:** Model selection criteria

### Clustering Metrics:
- **Silhouette Score**
- **Calinski-Harabasz Index**
- **Davies-Bouldin Index**
- **Adjusted Rand Index**
- **Normalized Mutual Information**

---

## Key Visualizations

1. **Covariance Ellipses:** Visualize cluster shapes
2. **Probability Heatmaps:** Soft clustering assignments
3. **Uncertainty Maps:** Points with ambiguous assignments
4. **BIC/AIC Curves:** Model selection plots
5. **Generated Samples:** GMM as generative model

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn plotly
```

---

## How to Run

1. Open `c_gaussian_mixture_models.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. The notebook generates synthetic data automatically

---

## Key Findings

- GMM outperforms K-Means on elliptical clusters
- Full covariance provides most flexibility
- BIC helps select optimal number of components
- Soft clustering provides uncertainty estimates

---

## Advantages over K-Means

| Feature | K-Means | GMM |
|---------|---------|-----|
| Cluster Shape | Spherical only | Elliptical |
| Assignment | Hard (0 or 1) | Soft (probabilities) |
| Uncertainty | No | Yes |
| Generative | No | Yes |

---

## Applications

- Clustering with uncertainty quantification
- Density estimation
- Anomaly detection (low likelihood points)
- Generative modeling (sample generation)

---

## References

- Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"
- Reynolds, D. (2009). "Gaussian Mixture Models"

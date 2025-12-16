# Time Series Clustering with Pretrained Models

## Assignment (f)

**Notebook:** `f_timeseries_clustering.ipynb`  
**Author:** Nitish  
**Date:** December 2024

---

## Overview

This notebook demonstrates **Time Series Clustering** using pretrained models and specialized distance metrics like **Dynamic Time Warping (DTW)**. It uses the TSlearn library for time series-specific clustering algorithms.

---

## Table of Contents

1. Introduction to Time Series Clustering
2. Dataset Preparation
3. Time Series Feature Extraction
4. Clustering with TSlearn
5. DTW-based Clustering
6. Stock Market Clustering
7. Evaluation Metrics

---

## Distance Metrics for Time Series

| Metric | Description | Best For |
|--------|-------------|----------|
| **Euclidean** | Point-to-point distance | Aligned series |
| **DTW** | Dynamic Time Warping | Temporal shifts |
| **Soft-DTW** | Differentiable DTW | Gradient-based optimization |

### Dynamic Time Warping (DTW):
- Allows elastic alignment of time series
- Handles temporal shifts and speed variations
- More robust than Euclidean for real-world data

---

## Features Implemented

- Euclidean K-Means for time series
- DTW K-Means clustering
- Soft-DTW K-Means
- Feature-based clustering
- Stock market time series clustering
- Barycenter (centroid) computation

---

## Datasets Used

| Dataset | Samples | Length | Classes |
|---------|---------|--------|---------|
| UCR Trace | 200 | 275 | 4 |
| Synthetic | 200 | 100 | 4 |
| Stock Returns | ~18 | 200 | - |

### Synthetic Time Series Classes:
1. Sine waves
2. Cosine waves
3. Sawtooth patterns
4. Random walks

---

## TSlearn Functions Used

```python
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import dtw, soft_dtw
from tslearn.barycenters import dtw_barycenter_averaging
```

---

## Feature-Based Approach

Statistical features extracted:
- Mean, Standard deviation
- Min, Max, Median
- Percentiles (25th, 75th)
- Total variation
- Argmax, Argmin
- Zero crossings

---

## Quality Metrics

### External Metrics:
- **Adjusted Rand Index (ARI)**
- **Normalized Mutual Information (NMI)**

### Internal Metrics:
- **Silhouette Score** (on feature space)

---

## Key Visualizations

1. **Time Series by Class:** Sample waveforms
2. **Cluster Centers:** DTW barycenters
3. **Stock Clustering:** Returns by sector
4. **Method Comparison:** Bar charts

---

## Requirements

```bash
pip install tslearn numpy pandas matplotlib seaborn scikit-learn yfinance
```

---

## How to Run

1. Open `f_timeseries_clustering.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. Stock data is downloaded automatically via yfinance

---

## Key Findings

- **DTW-based methods** outperform Euclidean for shifted series
- **Soft-DTW** provides differentiable alternative
- **Feature extraction** can be effective for simple patterns
- Stocks from same sector tend to cluster together

---

## Stock Market Clustering

### Stocks Analyzed:
- **Tech:** AAPL, MSFT, GOOGL, AMZN, META, NVDA
- **Finance:** JPM, BAC, WFC, GS
- **Energy:** XOM, CVX, COP, SLB
- **Healthcare:** JNJ, PFE, UNH, MRK

### Clustering Approach:
1. Download daily closing prices
2. Convert to daily returns
3. Normalize returns
4. Apply DTW K-Means

---

## Applications

- Stock market analysis
- Sensor data clustering
- ECG/medical signal analysis
- Speech pattern recognition
- Activity recognition

---

## When to Use DTW

✅ Time series have temporal shifts  
✅ Series have different speeds  
✅ Alignment is important  
✅ Shape similarity matters  

❌ Series are already aligned  
❌ Computational efficiency is critical  
❌ Very long time series  

---

## References

- Berndt, D. J., & Clifford, J. (1994). "Using Dynamic Time Warping to Find Patterns in Time Series"
- Cuturi, M., & Blondel, M. (2017). "Soft-DTW: a Differentiable Loss Function for Time-Series"
- TSlearn Documentation: https://tslearn.readthedocs.io/

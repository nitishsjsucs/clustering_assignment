# Anomaly Detection using PyOD

## Assignment (e)

**Notebook:** `e_anomaly_detection_pyod.ipynb`  
**Author:** Nitish  
**Date:** December 2024

---

## Overview

This notebook demonstrates **Anomaly Detection** (outlier detection) using the **PyOD** library. It covers univariate, multivariate, and time series anomaly detection with multiple algorithms and a real-world fraud detection use case.

---

## Table of Contents

1. Introduction to Anomaly Detection
2. PyOD Overview
3. Univariate Anomaly Detection
4. Multivariate Anomaly Detection
5. Time Series Anomaly Detection
6. Credit Card Fraud Detection Use Case
7. Model Comparison and Evaluation

---

## PyOD Algorithms Used

| Algorithm | Type | Description |
|-----------|------|-------------|
| **KNN** | Proximity | K-Nearest Neighbors based |
| **LOF** | Proximity | Local Outlier Factor |
| **IForest** | Ensemble | Isolation Forest |
| **OCSVM** | Linear | One-Class SVM |
| **HBOS** | Statistical | Histogram-based Outlier Score |
| **COPOD** | Probabilistic | Copula-Based Outlier Detection |
| **ECOD** | Probabilistic | Empirical Cumulative Distribution |

---

## Features Implemented

- Univariate anomaly detection
- Multivariate anomaly detection (3+ features)
- Time series anomaly detection with feature engineering
- Credit card fraud detection simulation
- Multiple algorithm comparison
- ROC-AUC and Precision-Recall evaluation

---

## Datasets Used

| Dataset | Type | Samples | Contamination |
|---------|------|---------|---------------|
| Synthetic 2D | Multivariate | 500 | 10% |
| Univariate | 1D | 1000 | 10% |
| Time Series | Temporal | 1000 | 5% |
| Credit Card | Fraud | 5000 | 2% |

---

## Time Series Feature Engineering

Features extracted from sliding window:
- Current value
- Deviation from window mean
- Z-score
- Change from previous point
- Window standard deviation

---

## Quality Metrics

### Ranking Metrics:
- **ROC-AUC:** Area under ROC curve (0-1, higher is better)
- **Average Precision:** Area under PR curve

### Classification Metrics:
- **Precision:** True positives / Predicted positives
- **Recall:** True positives / Actual positives
- **F1-Score:** Harmonic mean of precision and recall

---

## Key Visualizations

1. **Scatter Plots:** Normal vs anomaly points
2. **ROC Curves:** Model comparison
3. **Precision-Recall Curves:** For imbalanced data
4. **Time Series Plots:** Anomaly scores over time
5. **Confusion Matrices:** Classification results

---

## Requirements

```bash
pip install pyod numpy pandas matplotlib seaborn scikit-learn
```

---

## How to Run

1. Open `e_anomaly_detection_pyod.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. The notebook generates synthetic data automatically

---

## Key Findings

- **Isolation Forest** and **COPOD** generally perform well
- **LOF** is effective for local anomalies
- **ECOD** is fast and parameter-free
- Feature engineering is crucial for time series

---

## Use Case: Credit Card Fraud Detection

### Features Used:
- Transaction amount
- Time since last transaction
- Distance from home

### Fraud Patterns:
- Higher transaction amounts
- Quick succession of transactions
- Far from usual locations

---

## Best Practices

1. **Standardize features** before detection
2. **Set contamination rate** based on domain knowledge
3. **Use ROC-AUC** for ranking evaluation
4. **Use Average Precision** for imbalanced data
5. **Compare multiple models** before selecting

---

## When to Use Each Algorithm

| Scenario | Recommended |
|----------|-------------|
| General purpose | IForest, COPOD |
| Local anomalies | LOF |
| Fast detection | HBOS, ECOD |
| High-dimensional | IForest |
| Small datasets | KNN, LOF |

---

## References

- Zhao, Y., et al. (2019). "PyOD: A Python Toolbox for Scalable Outlier Detection"
- Liu, F. T., et al. (2008). "Isolation Forest"
- Breunig, M. M., et al. (2000). "LOF: Identifying Density-Based Local Outliers"

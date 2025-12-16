# Audio Clustering with Deep Learning Embeddings

## Assignment (i)

**Notebook:** `i_audio_clustering_embeddings.ipynb`  
**Author:** Nitish  
**Date:** December 2024

---

## Overview

This notebook demonstrates **Audio Clustering** using deep learning embeddings extracted from audio signals. It uses Librosa for audio feature extraction (MFCCs, spectral features) and applies clustering algorithms to group similar audio signals.

---

## Table of Contents

1. Introduction
2. Setup and Installation
3. Audio Feature Extraction
4. Generate Audio Dataset
5. Extract Audio Embeddings
6. Clustering Analysis
7. Visualization
8. Evaluation Metrics

---

## Audio Features Extracted

### MFCC (Mel-Frequency Cepstral Coefficients):
- 13 MFCCs (mean and std)
- Captures timbral characteristics
- Most important for audio classification

### Spectral Features:
| Feature | Description |
|---------|-------------|
| Spectral Centroid | "Brightness" of sound |
| Spectral Bandwidth | Width of frequency band |
| Spectral Rolloff | Frequency below which 85% energy |
| Zero Crossing Rate | Rate of sign changes |
| RMS Energy | Loudness measure |
| Chroma | Pitch class distribution |

---

## Features Implemented

- Synthetic audio signal generation
- Comprehensive audio feature extraction
- Spectrogram visualization
- PCA and UMAP dimensionality reduction
- K-Means and Hierarchical clustering
- Cluster analysis and evaluation

---

## Dataset

### Synthetic Audio Classes (250 samples total):

| Class | Type | Frequency Range |
|-------|------|-----------------|
| 0 | Low Sine Wave | 200-400 Hz |
| 1 | High Sine Wave | 800-1200 Hz |
| 2 | Square Wave | 300-600 Hz |
| 3 | Chirp (Sweep) | 200→1200 Hz |
| 4 | White Noise | Broadband |

### Audio Parameters:
- **Sample Rate:** 22,050 Hz
- **Duration:** 1 second
- **Samples per class:** 50

---

## Workflow

```
Audio Signal → Librosa Features → Feature Vector (46D)
                                        ↓
                                   StandardScaler
                                        ↓
                                   PCA (20D)
                                        ↓
                                   UMAP (2D)
                                        ↓
                                   K-Means
```

---

## Feature Extraction Code

```python
def extract_audio_features(signal, sr):
    features = []
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    features.append(np.mean(spectral_centroid))
    
    # ... more features
    return np.array(features)
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

1. **Waveforms:** Sample audio signals
2. **Spectrograms:** Frequency content over time
3. **UMAP Projection:** 2D feature space
4. **Elbow Plot:** Optimal K selection
5. **Confusion Matrix:** Class vs cluster
6. **Dendrogram:** Hierarchical relationships

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn librosa soundfile umap-learn
```

---

## How to Run

1. Open `i_audio_clustering_embeddings.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. Audio signals are generated synthetically

---

## Key Findings

- MFCC and spectral features effectively capture audio characteristics
- Different audio types form distinct clusters
- Frequency-based features are most discriminative
- K-Means achieves high clustering accuracy

---

## Audio Type Characteristics

| Type | Spectral Centroid | Zero Crossings | Energy |
|------|-------------------|----------------|--------|
| Low Sine | Low | Low | Stable |
| High Sine | High | High | Stable |
| Square | Medium | Very High | Stable |
| Chirp | Varying | Varying | Stable |
| Noise | Medium | Very High | Variable |

---

## Applications

- Music genre classification
- Speaker identification
- Sound event detection
- Audio content organization
- Environmental sound classification
- Musical instrument recognition

---

## Alternative Approaches

| Method | Description |
|--------|-------------|
| VGGish | CNN trained on AudioSet |
| OpenL3 | Self-supervised audio embeddings |
| wav2vec | Speech representation learning |
| CLAP | Audio-text contrastive learning |

---

## Librosa Functions Used

```python
import librosa

# Load audio
y, sr = librosa.load(audio_file)

# Extract features
mfccs = librosa.feature.mfcc(y=y, sr=sr)
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
rms = librosa.feature.rms(y=y)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Spectrogram
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D))
```

---

## References

- McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python"
- Davis, S., & Mermelstein, P. (1980). "Comparison of Parametric Representations for Monosyllabic Word Recognition"
- Librosa Documentation: https://librosa.org/

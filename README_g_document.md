# Document Clustering with LLM Embeddings

## Assignment (g)

**Notebook:** `g_document_clustering_llm.ipynb`  
**Author:** Nitish  
**Date:** December 2024

---

## Overview

This notebook demonstrates **Document Clustering** using state-of-the-art **LLM embeddings** from Sentence Transformers. It shows how to convert text documents into semantic vector representations and cluster them based on meaning.

---

## Table of Contents

1. Introduction
2. Dataset Preparation
3. Sentence Transformers Embeddings
4. Document Clustering
5. Visualization with UMAP
6. Topic Analysis
7. Evaluation Metrics

---

## Embedding Model

### Sentence Transformers:
- **Model:** `all-MiniLM-L6-v2`
- **Embedding Dimension:** 384
- **Type:** Pre-trained transformer model
- **Advantage:** Captures semantic meaning, not just keywords

### How It Works:
1. Input text is tokenized
2. Passed through transformer encoder
3. Pooled to fixed-size vector
4. Semantically similar texts have similar vectors

---

## Features Implemented

- Document embedding generation
- Cosine similarity matrix computation
- UMAP dimensionality reduction
- K-Means and Hierarchical clustering
- Cluster composition analysis
- Representative document extraction

---

## Dataset

### 5 Document Categories (50 documents total):

| Category | Topics | Documents |
|----------|--------|-----------|
| **Technology** | AI, ML, Cloud, Cybersecurity | 10 |
| **Sports** | Football, Basketball, Tennis | 10 |
| **Science** | Climate, DNA, Space | 10 |
| **Finance** | Stocks, Crypto, Banking | 10 |
| **Health** | Exercise, Diet, Mental Health | 10 |

---

## Workflow

```
Documents → Sentence Transformer → Embeddings (384D)
                                        ↓
                                   UMAP (2D)
                                        ↓
                                   K-Means
                                        ↓
                                   Clusters
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

1. **Similarity Matrix:** Document-to-document cosine similarity
2. **UMAP Projection:** 2D visualization of embeddings
3. **Elbow Plot:** Optimal K selection
4. **Cluster Comparison:** True vs predicted labels
5. **Dendrogram:** Hierarchical relationships
6. **Confusion Matrix:** Category vs cluster mapping

---

## Requirements

```bash
pip install sentence-transformers numpy pandas matplotlib seaborn scikit-learn umap-learn
```

---

## How to Run

1. Open `g_document_clustering_llm.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. Documents are defined in the notebook

---

## Key Findings

- Sentence Transformers capture semantic similarity effectively
- Documents about similar topics cluster together
- UMAP provides clear visualization of document relationships
- K-Means achieves high agreement with true categories

---

## Embedding Models Comparison

| Model | Speed | Quality | Dimension |
|-------|-------|---------|-----------|
| all-MiniLM-L6-v2 | Fast | Good | 384 |
| paraphrase-MiniLM-L6-v2 | Fast | Good | 384 |
| all-mpnet-base-v2 | Medium | Best | 768 |

---

## Applications

- Document organization and filing
- Topic discovery in large corpora
- Content recommendation systems
- Search result clustering
- Duplicate detection
- News article grouping

---

## Advantages of LLM Embeddings

| Traditional (TF-IDF) | LLM Embeddings |
|---------------------|----------------|
| Keyword matching | Semantic understanding |
| Sparse vectors | Dense vectors |
| No context | Contextual |
| Synonyms ignored | Synonyms captured |

---

## Code Example

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# Cluster
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(embeddings)
```

---

## References

- Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- Sentence Transformers: https://www.sbert.net/
- UMAP: McInnes, L., et al. (2018). "UMAP: Uniform Manifold Approximation and Projection"

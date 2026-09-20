# Clustering assignment

Nine notebooks (a–i) covering one clustering technique each, written for a graduate course
assignment in December 2024. Each notebook is self-contained: it installs what it needs, generates
or downloads its own data, runs the method, and scores the result with silhouette /
Calinski-Harabasz / Davies-Bouldin plus ARI and NMI where labels exist.

The notebooks are committed **without outputs** — no plots or scores are visible on GitHub, so run
one to see results. Each has a companion `README_<letter>.md` with a longer write-up.

## What's here

| Notebook | Method | Data |
|---|---|---|
| `a_kmeans_from_scratch.ipynb` | K-Means written from scratch, including K-Means++ init, elbow and silhouette analysis, checked against scikit-learn | `make_blobs`, iris, wine |
| `b_hierarchical_clustering.ipynb` | Agglomerative clustering, dendrograms, single/complete/average/ward linkage compared | `make_blobs`, `make_moons`, iris, wine |
| `c_gaussian_mixture_models.ipynb` | GMM with different covariance types, soft assignments, BIC/AIC selection, vs. K-Means | `make_blobs`, iris, wine |
| `d_dbscan_pycaret.ipynb` | DBSCAN through PyCaret, eps tuning via a k-distance graph, on non-convex shapes | `make_moons`, `make_circles`, `make_blobs`, iris |
| `e_anomaly_detection_pyod.ipynb` | PyOD detectors side by side — IForest, LOF, KNN, OCSVM, HBOS, COPOD, ECOD — over univariate, multivariate, time-series and a card-fraud scenario, scored with ROC-AUC and average precision | all synthetic: PyOD's `generate_data`, plus lognormal/exponential "transactions" built in the notebook |
| `f_timeseries_clustering.ipynb` | DTW and soft-DTW k-means with `tslearn`, plus feature-based clustering | the UCR "Trace" dataset via `tslearn`, and stock series pulled with `yfinance` |
| `g_document_clustering_llm.ipynb` | Sentence-Transformers embeddings (`all-MiniLM-L6-v2`, `paraphrase-MiniLM-L6-v2`), UMAP projection, then K-Means and hierarchical clustering | ~50 short sentences written inline across five topics |
| `h_image_clustering_imagebind.ipynb` | Image embeddings clustered and evaluated against the true labels, with PCA/UMAP views | CIFAR-10 via `torchvision` |
| `i_audio_clustering_embeddings.ipynb` | MFCC and spectral features from `librosa`, spectrograms, K-Means and hierarchical clustering | synthetic waveforms generated in the notebook (sine, square, chirp, noise) |

Two things the filenames oversell: notebook `h` is named for ImageBind but actually extracts
features from a `torchvision` **ResNet50** pretrained on ImageNet — the notebook says as much, calling
them "ImageBind-style" embeddings. And the "LLM embeddings" in `g` are sentence-transformer
embeddings of a small hand-written corpus, not a real document collection.

## Running it

Open any notebook in Colab or Jupyter and run it top to bottom. Each starts with its own `pip
install` line; across all nine that adds up to:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy plotly
pip install pycaret pyod tslearn sentence-transformers umap-learn
pip install torch torchvision librosa soundfile yfinance
```

Notebooks `g` and `h` download pretrained model weights, `f` hits the network for stock data, and
`h` downloads CIFAR-10. The rest run offline on CPU in a minute or two.

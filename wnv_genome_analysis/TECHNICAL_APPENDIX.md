# Technical Appendix: Detailed Statistical Analysis

## A1. Feature Engineering Details

### K-mer Analysis
- **3-mers**: 4³ = 64 possible combinations
- **4-mers**: 4⁴ = 256 possible combinations
- **Frequency normalization**: f(kmer) = count(kmer) / total_kmers
- **Missing k-mers**: Assigned frequency = 0

### Standardization Method
- **Algorithm**: StandardScaler (sklearn)
- **Formula**: z = (x - μ) / σ
- **Post-standardization**: Mean = 0, Std = 1 for all features

## A2. Statistical Test Results

### A2.1 Clustering Validation Metrics - Detailed Results

```
K=2: Silhouette=0.554, Calinski-Harabasz=1863.2, Davies-Bouldin=0.840
K=3: Silhouette=0.531, Calinski-Harabasz=1180.9, Davies-Bouldin=1.449
K=4: Silhouette=0.514, Calinski-Harabasz=964.7, Davies-Bouldin=1.327
K=5: Silhouette=0.280, Calinski-Harabasz=843.9, Davies-Bouldin=1.440
K=6: Silhouette=0.283, Calinski-Harabasz=797.3, Davies-Bouldin=1.370
K=7: Silhouette=0.285, Calinski-Harabasz=735.4, Davies-Bouldin=1.138
K=8: Silhouette=0.265, Calinski-Harabasz=695.3, Davies-Bouldin=1.324
```

### A2.2 Sub-cluster Analysis Results

**Main Cluster 0 (n=1581)**:
```
K=2: Silhouette=0.493, CH=369.3, DB=1.350  ← OPTIMAL
K=3: Silhouette=0.225, CH=369.7, DB=1.514
K=4: Silhouette=0.201, CH=337.4, DB=1.779
K=5: Silhouette=0.208, CH=299.8, DB=1.565
K=6: Silhouette=0.211, CH=263.3, DB=1.732
```

**Main Cluster 1 (n=487)**:
```
K=2: Silhouette=0.493, CH=146.5, DB=1.473
K=3: Silhouette=0.479, CH=156.6, DB=1.171
K=4: Silhouette=0.485, CH=152.9, DB=0.762
K=5: Silhouette=0.499, CH=163.8, DB=0.737
K=6: Silhouette=0.505, CH=153.2, DB=0.943  ← OPTIMAL
```

## A3. Algorithm Parameters

### K-means Parameters
- **n_init**: 20 (multiple random initializations)
- **random_state**: 42 (reproducibility)
- **algorithm**: 'lloyd' (default)
- **max_iter**: 300 (default)

### PCA Parameters
- **n_components**: 2 (for visualization)
- **whiten**: False
- **random_state**: 42

### UMAP Parameters
- **n_neighbors**: 15 (or min(15, n_samples-1))
- **n_components**: 2
- **metric**: 'euclidean'
- **random_state**: 42

## A4. Statistical Significance Tests

### Silhouette Score Interpretation
- **> 0.7**: Excellent clustering
- **0.5-0.7**: Good clustering  ✓ Our K=2 result: 0.554
- **0.25-0.5**: Weak clustering
- **< 0.25**: Poor clustering

### Calinski-Harabasz Score
- **Higher values indicate better separation**
- **Our K=2 score**: 1863.2 (peak value)
- **Declining trend** for K>2 indicates overfitting

### Davies-Bouldin Score
- **Lower values indicate better clustering**
- **Our K=2 score**: 0.840 (< 1.0 = good)
- **Threshold**: < 1.0 for acceptable clustering ✓

## A5. Biological Validation

### Known WNV Lineage Structure
- **Lineage 1**: Global distribution, multiple clades
- **Lineage 2**: African origin, European spread
- **Literature consensus**: 2 major lineages with sub-clades

### Our Results Validation
- **Statistical optimal**: K=2 ✓
- **Biological match**: 2 major lineages ✓
- **Sub-structure**: 2+6=8 total sub-lineages ✓

## A6. Reproducibility Information

### Random Seeds Used
- **K-means**: 42
- **PCA**: 42  
- **UMAP**: 42
- **t-SNE**: 42

### Software Versions
- **Python**: 3.13
- **scikit-learn**: Latest
- **BioPython**: Latest
- **UMAP**: Latest
- **NumPy**: Latest
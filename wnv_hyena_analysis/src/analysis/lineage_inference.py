"""Lineage inference for West Nile Virus sequences.

Unsupervised + semi-supervised approach:
 - Dimensionality reduction (PCA) for noise reduction
 - Multiple clustering algorithms (KMeans, Agglomerative) over range of k
 - Model selection via silhouette / Davies-Bouldin / Calinski-Harabasz composite score
 - Optional incorporation of existing sparse lineage labels as seeds (purity-based naming)
 - Statistical association tests (geography, host) with chi-square + Cramer's V
 - Outlier detection (IsolationForest + distance to centroid) consolidated
 - Outputs comprehensive JSON-ready dictionary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.ensemble import IsolationForest
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.stats import chi2_contingency

from src.utils.logger import setup_logger

# Optional algorithms
try:
    import hdbscan  # type: ignore
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

try:
    import umap  # type: ignore
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False


@dataclass
class LineageInferenceConfig:
    min_k: int = 2
    max_k: int = 20
    max_k_fraction: float = 0.25  # max k limited by n_samples * fraction
    pca_variance: float = 0.95
    random_state: int = 42
    min_cluster_size: int = 5
    lineage_purity_threshold: float = 0.7
    bootstrap_reps: int = 10
    outlier_contamination: float = 0.02
    use_umap: bool = True
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    # Advanced merging constraints
    merge_max_lineages: int = 9          # target upper bound for lineage count
    merge_ratio_threshold: float = 0.9    # centroid distance / (radiusA+radiusB) below this => candidate merge
    merge_percentile_threshold: float = 0.15  # centroid distance percentile threshold for merging
    distance_sample_cap: int = 800        # sample size cap for pairwise distance computations


class LineageInferer:
    def __init__(self, config: dict | None = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger("lineage_inference")
        cfg_section = (config or {}).get("lineage_inference", {})
        self.cfg = LineageInferenceConfig(**{k: v for k, v in cfg_section.items() if k in LineageInferenceConfig.__annotations__})
        self.results: Dict[str, Any] = {}

    # ----------------------- Public API -----------------------
    def infer(self, features: np.ndarray, df_meta: pd.DataFrame) -> Dict[str, Any]:
        """Main lineage inference workflow."""
        n_samples = features.shape[0]
        if n_samples < self.cfg.min_k * 2:
            self.logger.warning("Not enough samples for lineage inference; skipping")
            return {"skipped": True, "reason": "insufficient_samples"}

        self.logger.info(
            f"Starting lineage inference on {n_samples} samples, {features.shape[1]} features"
        )

        # 1. Dimensionality reduction
        X_red, pca_model = self._pca_reduce(features)
        X_embed = self._umap_embed(X_red) if (self.cfg.use_umap and UMAP_AVAILABLE) else X_red

        # 2. Clustering model selection
        best_model_info = self._select_clustering(X_embed)
        labels = best_model_info["model"].fit_predict(X_embed)

        # 3. Initial cluster statistics
        cluster_stats = self._compute_cluster_stats(X_embed, labels)

        # 4. Merge clusters statistically to enforce max lineage count
        labels, merge_history, cluster_stats = self._merge_lineages(X_embed, labels, cluster_stats)

        # 5. Assign human-readable lineage names (leveraging any existing labels)
        lineage_assignments, lineage_mapping = self._assign_lineage_names(labels, df_meta)

        # 6. Outlier detection
        outlier_flags, outlier_scores = self._detect_outliers(X_embed, labels, cluster_stats)

        # 7. Category association tests
        assoc_geo = self._association_test(lineage_assignments, df_meta.get("country"))
        assoc_host = self._association_test(lineage_assignments, df_meta.get("host"))

        # 8. Hierarchical relationship tree (centroid-based)
        tree_newick = self._build_newick(X_embed, labels)

        # 9. Additional validation metrics (inter/intra distances, silhouettes)
        validation_metrics = self._compute_validation_metrics(X_embed, labels, cluster_stats)
        self.results = {
            "n_samples": int(n_samples),
            "pca_components": int(X_red.shape[1]),
            "umap_used": bool(self.cfg.use_umap and UMAP_AVAILABLE),
            "embedding_dim": int(X_embed.shape[1]),
            "explained_variance_ratio": pca_model.explained_variance_ratio_.cumsum().tolist(),
            "selected_model": {
                "algorithm": best_model_info["algorithm"],
                "k": best_model_info.get("k"),
                "selection_score": best_model_info.get("selection_score"),
                "internal_metrics": best_model_info.get("metrics"),
            },
            "cluster_stats": cluster_stats,
            "lineage_mapping": lineage_mapping,
            "lineage_labels": lineage_assignments.tolist(),
            "merge_history": merge_history,
            "outliers": {
                "flags": outlier_flags.tolist(),
                "scores": outlier_scores.tolist(),
                "count": int(outlier_flags.sum()),
            },
            "associations": {"geography": assoc_geo, "host": assoc_host},
            "newick": tree_newick,
            "validation_metrics": validation_metrics,
        }
        return self.results

    # ------------------- Internal helpers --------------------
    def _pca_reduce(self, X: np.ndarray) -> Tuple[np.ndarray, PCA]:
        pca = PCA(n_components=min(X.shape[0]-1, X.shape[1]), random_state=self.cfg.random_state)
        X_pca = pca.fit_transform(X)
        # Determine number of components reaching variance threshold
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_comp = np.searchsorted(cumvar, self.cfg.pca_variance) + 1
        X_red = X_pca[:, :n_comp]
        self.logger.info(f"PCA reduced features to {n_comp} components (>= {self.cfg.pca_variance*100:.1f}% variance)")
        return X_red, pca

    def _select_clustering(self, X: np.ndarray) -> Dict[str, Any]:
        n = X.shape[0]
        max_k_allowed = max(self.cfg.min_k + 1, min(self.cfg.max_k, int(n * self.cfg.max_k_fraction)))
        candidate_ks = list(range(self.cfg.min_k, max_k_allowed + 1))
        algorithms = ["kmeans", "agglomerative"]
        if HDBSCAN_AVAILABLE:
            algorithms.append("hdbscan")
        best = None
        best_score = -np.inf
        self.logger.info(f"Evaluating clustering models for k in {candidate_ks}")
        for algo in algorithms:
            for k in candidate_ks:
                try:
                    if algo == "hdbscan":
                        # For HDBSCAN we ignore k loop and run once with min_cluster_size heuristic
                        if not HDBSCAN_AVAILABLE or k != candidate_ks[0]:
                            continue
                        model = hdbscan.HDBSCAN(min_cluster_size=max(5, int(0.01 * n)),
                                                metric='euclidean',
                                                core_dist_n_jobs=1,
                                                random_state=self.cfg.random_state if hasattr(hdbscan.HDBSCAN, 'random_state') else None)
                        labels = model.fit_predict(X)
                        if len(set(labels)) <= 1:
                            continue
                        # Remove noise label (-1) for metrics
                        mask = labels != -1
                        if mask.sum() < 5:
                            continue
                        sil = silhouette_score(X[mask], labels[mask])
                        # Davies-Bouldin not defined with noise; approximate with remaining
                        db = davies_bouldin_score(X[mask], labels[mask])
                        ch = calinski_harabasz_score(X[mask], labels[mask])
                        composite = sil + (ch / 1000.0) - db
                        if composite > best_score:
                            best_score = composite
                            best = {"algorithm": algo, "k": int(len(set(labels)) - (1 if -1 in labels else 0)),
                                    "model": model, "selection_score": composite,
                                    "metrics": {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}}
                        continue
                    if k >= n:
                        continue
                    model = (KMeans(n_clusters=k, random_state=self.cfg.random_state, n_init='auto')
                             if algo == "kmeans" else AgglomerativeClustering(n_clusters=k))
                    labels = model.fit_predict(X)
                    if len(set(labels)) < 2:
                        continue
                    sil = silhouette_score(X, labels)
                    db = davies_bouldin_score(X, labels)
                    ch = calinski_harabasz_score(X, labels)
                    # Composite score: higher silhouette & CH, lower DB
                    composite = sil + (ch / 1000.0) - db
                    if composite > best_score:
                        best_score = composite
                        best = {
                            "algorithm": algo,
                            "k": k,
                            "model": model,
                            "selection_score": composite,
                            "metrics": {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
                        }
                except Exception as e:
                    self.logger.warning(f"Clustering failed for {algo} k={k}: {e}")
        if best is None:
            raise RuntimeError("Failed to select clustering model")
        self.logger.info(f"Selected {best['algorithm']} with k={best['k']} (score={best['selection_score']:.3f})")
        return best

    def _umap_embed(self, X: np.ndarray) -> np.ndarray:
        if not UMAP_AVAILABLE:
            return X
        self.logger.info(f"Applying UMAP embedding (n_neighbors={self.cfg.umap_n_neighbors}, min_dist={self.cfg.umap_min_dist})")
        reducer = umap.UMAP(n_neighbors=min(self.cfg.umap_n_neighbors, X.shape[0]-1),
                            min_dist=self.cfg.umap_min_dist,
                            random_state=self.cfg.random_state,
                            metric='euclidean')
        X_umap = reducer.fit_transform(X)
        self.logger.info(f"UMAP produced embedding dimension: {X_umap.shape[1]}")
        return X_umap

    def _compute_cluster_stats(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        stats = {}
        for cluster in sorted(set(labels)):
            idx = np.where(labels == cluster)[0]
            subset = X[idx]
            centroid = subset.mean(axis=0)
            dists = np.linalg.norm(subset - centroid, axis=1)
            stats[str(cluster)] = {
                "size": int(len(idx)),
                "centroid": centroid.tolist(),
                "radius_mean": float(dists.mean()),  # mean Euclidean distance to centroid
                "radius_std": float(dists.std()),
                "radius_q3": float(np.quantile(dists, 0.75)),
                "dispersion": float(np.mean(dists**2))  # mean squared distance
            }
        return stats

    def _merge_lineages(self, X: np.ndarray, labels: np.ndarray, cluster_stats: Dict[str, Any]):
        """Iteratively merge highly similar clusters until constraints satisfied.

        Similarity criteria:
          1. Centroid distance percentile < merge_percentile_threshold among all centroid distances.
          2. Normalized distance ratio d_AB / (radiusA + radiusB) < merge_ratio_threshold.
        Produces merge history with statistical context (percentile, ratio, raw distance).
        """
        unique_clusters = sorted(set(labels))
        if len(unique_clusters) <= self.cfg.merge_max_lineages:
            return labels, [], cluster_stats

        # Precompute centroids and radii
        centroids = {c: np.array(cluster_stats[str(c)]["centroid"]) for c in unique_clusters}
        radii = {c: cluster_stats[str(c)]["radius_mean"] for c in unique_clusters}

        # Distance matrix between centroids
        def centroid_distance(a, b):
            return float(np.linalg.norm(centroids[a] - centroids[b]))

        merge_history = []

        while True:
            current = sorted(set(labels))
            if len(current) <= self.cfg.merge_max_lineages:
                break
            # Compute all pair stats
            pair_stats = []
            distances = []
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    a, b = current[i], current[j]
                    d = centroid_distance(a, b)
                    distances.append(d)
                    ratio = d / max(1e-9, (radii[a] + radii[b]))
                    pair_stats.append((a, b, d, ratio))
            if not pair_stats:
                break
            # Compute distance percentiles
            distances_arr = np.array(distances)
            # Evaluate candidate pairs
            candidates = []
            for a, b, d, ratio in pair_stats:
                pct = float((distances_arr < d).sum() / len(distances_arr))
                if pct <= self.cfg.merge_percentile_threshold and ratio <= self.cfg.merge_ratio_threshold:
                    # Statistical separation surrogate: z = d / std(distances)
                    z = (d - distances_arr.mean()) / (distances_arr.std() + 1e-9)
                    candidates.append((ratio, d, pct, z, a, b))
            if not candidates:
                break
            # Select strongest merge (lowest ratio then distance)
            candidates.sort()
            ratio, d, pct, z, a, b = candidates[0]
            # Merge b into a
            labels = np.array([a if lab == b else lab for lab in labels])
            # Recompute stats for merged cluster a
            idx_a = np.where(labels == a)[0]
            subset = X[idx_a]
            centroid = subset.mean(axis=0)
            dists = np.linalg.norm(subset - centroid, axis=1)
            cluster_stats[str(a)] = {
                "size": int(len(idx_a)),
                "centroid": centroid.tolist(),
                "radius_mean": float(dists.mean()),
                "radius_std": float(dists.std()),
                "radius_q3": float(np.quantile(dists, 0.75)),
                "dispersion": float(np.mean(dists**2))
            }
            # Remove old cluster stat for b
            if str(b) in cluster_stats:
                del cluster_stats[str(b)]
            # Update centroids/radii maps
            centroids[a] = centroid
            radii[a] = cluster_stats[str(a)]["radius_mean"]
            if b in centroids:
                del centroids[b]
                del radii[b]
            merge_history.append({
                "merge_pair": [int(a), int(b)],
                "centroid_distance": d,
                "distance_percentile": pct,
                "distance_zscore": z,
                "ratio": ratio,
                "remaining_lineages": len(set(labels))
            })
        # Relabel clusters to consecutive integers for cleanliness
        unique_final = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique_final)}
        labels = np.array([remap[l] for l in labels])
        # Rebuild stats dict with new keys
        new_stats = {}
        for old, new in remap.items():
            new_stats[str(new)] = cluster_stats[str(old)]
        return labels, merge_history, new_stats

    def _assign_lineage_names(self, labels: np.ndarray, df_meta: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        existing = df_meta.get('lineage') if 'lineage' in df_meta.columns else None
        cluster_lineage = {}
        lineage_names = []
        if existing is not None and existing.notna().any():
            existing = existing.fillna('Unknown')
        for cluster in sorted(set(labels)):
            idx = np.where(labels == cluster)[0]
            majority_label = None
            purity = 0.0
            if existing is not None:
                vals = existing.iloc[idx]
                # Remove NaNs and placeholder Unknown
                vals = vals[vals.notna() & (vals != 'Unknown')]
                if len(vals) >= 3:
                    counts = vals.value_counts()
                    if not counts.empty:
                        top_label, top_count = counts.index[0], counts.iloc[0]
                        purity = top_count / len(vals)
                        if purity >= self.cfg.lineage_purity_threshold:
                            majority_label = top_label.upper()
            if majority_label is None:
                majority_label = f"LINEAGE_{cluster}"
            cluster_lineage[str(cluster)] = {"assigned": majority_label, "purity": purity}
            lineage_names.append(majority_label)
        # Map cluster label to lineage string array
        lineage_assignments = np.array([cluster_lineage[str(c)]["assigned"] for c in labels])
        return lineage_assignments, cluster_lineage

    def _compute_validation_metrics(self, X: np.ndarray, labels: np.ndarray, cluster_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Compute inter/intra lineage distance metrics and silhouettes.

        Returns keys:
          - centroid_distance_matrix
          - dunn_index
          - per_lineage (size, radius_mean, nearest_centroid_distance, separation_ratio)
          - silhouette_per_lineage / silhouette_overall (if >1 cluster)
          - summary (min_inter_centroid, max_intra_radius, avg_radius, avg_nearest_centroid)
        """
        unique = sorted(set(labels))
        if len(unique) == 0:
            return {"available": False}
        # Build centroid matrix
        centroids = {c: np.array(cluster_stats[str(c)]["centroid"]) for c in unique}
        k = len(unique)
        dist_mat = np.zeros((k, k), dtype=float)
        for i, a in enumerate(unique):
            for j, b in enumerate(unique):
                if j <= i:
                    continue
                d = float(np.linalg.norm(centroids[a] - centroids[b]))
                dist_mat[i, j] = dist_mat[j, i] = d
        # Per-lineage nearest neighbor distance
        nearest = {}
        for i, a in enumerate(unique):
            others = [dist_mat[i, j] for j in range(k) if j != i and dist_mat[i, j] > 0]
            nearest[a] = float(min(others)) if others else float('inf')
        # Dunn index components
        inter_min = float(np.min(dist_mat[np.triu_indices(k, 1)]) if k > 1 else 0.0)
        intra_max = 0.0
        per_lineage = {}
        radii = []
        sep_ratios = []
        for c in unique:
            r = cluster_stats[str(c)]["radius_mean"]
            intra_max = max(intra_max, 2 * r)  # approximate diameter ~ 2 * mean radius
            nn = nearest[c]
            sep_ratio = float(nn / (r + 1e-9)) if np.isfinite(nn) else float('inf')
            per_lineage[str(c)] = {
                "size": cluster_stats[str(c)]["size"],
                "radius_mean": r,
                "nearest_centroid_distance": nn,
                "separation_ratio": sep_ratio,
            }
            radii.append(r)
            if np.isfinite(sep_ratio):
                sep_ratios.append(sep_ratio)
        dunn = float(inter_min / (intra_max + 1e-9)) if intra_max > 0 and k > 1 else None
        # Silhouette metrics
        silhouette_overall = None
        silhouette_per_lineage = {}
        if k > 1 and len(X) > k:
            try:
                sil_samples = silhouette_samples(X, labels)
                silhouette_overall = float(np.mean(sil_samples))
                for c in unique:
                    silhouette_per_lineage[str(c)] = float(np.mean(sil_samples[labels == c]))
            except Exception:
                pass
        summary = {
            "min_inter_centroid_distance": inter_min if k > 1 else None,
            "max_intra_radius_mean": float(max(radii)) if radii else None,
            "dunn_index": dunn,
            "avg_radius_mean": float(np.mean(radii)) if radii else None,
            "avg_nearest_centroid_distance": float(np.mean([nearest[c] for c in unique if np.isfinite(nearest[c])])) if k > 1 else None,
            "median_separation_ratio": float(np.median(sep_ratios)) if sep_ratios else None,
        }
        return {
            "available": True,
            "centroid_distance_matrix": dist_mat.tolist(),
            "per_lineage": per_lineage,
            "silhouette_overall": silhouette_overall,
            "silhouette_per_lineage": silhouette_per_lineage,
            "summary": summary,
        }

    def _detect_outliers(self, X: np.ndarray, labels: np.ndarray, cluster_stats: Dict[str, Any]):
        iso = IsolationForest(contamination=self.cfg.outlier_contamination, random_state=self.cfg.random_state)
        iso.fit(X)
        iso_scores = -iso.decision_function(X)  # higher = more anomalous
        flags_iso = iso.predict(X) == -1
        # Distance-based flags
        dist_flags = np.zeros(len(X), dtype=bool)
        for cluster in set(labels):
            idx = np.where(labels == cluster)[0]
            centroid = np.array(cluster_stats[str(cluster)]["centroid"])
            dists = np.linalg.norm(X[idx] - centroid, axis=1)
            q3 = cluster_stats[str(cluster)]["radius_q3"]
            std = cluster_stats[str(cluster)]["radius_std"]
            thresh = q3 + 1.5 * std
            dist_flags[idx] = dists > thresh
        combined_flags = flags_iso | dist_flags
        return combined_flags, iso_scores

    def _association_test(self, lineage_labels: np.ndarray, category_series: Optional[pd.Series]):
        if category_series is None:
            return {"available": False}
        series = category_series.fillna('Unknown')
        valid_mask = series != 'Unknown'
        if valid_mask.sum() < 5:
            return {"available": False}
        table = pd.crosstab(lineage_labels[valid_mask], series[valid_mask])
        if table.shape[0] < 2 or table.shape[1] < 2:
            return {"available": False}
        chi2, p, dof, expected = chi2_contingency(table.values)
        # Cramer's V
        n = table.values.sum()
        phi2 = chi2 / n
        r, k = table.shape
        cramers_v = np.sqrt(phi2 / max((min(k - 1, r - 1)), 1))
        return {
            "available": True,
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "cramers_v": float(cramers_v),
            "table": table.to_dict()
        }

    def _build_newick(self, X: np.ndarray, labels: np.ndarray) -> str:
        # Use cluster centroids for hierarchical tree
        centroids = []
        cluster_ids = sorted(set(labels))
        for c in cluster_ids:
            centroids.append(X[labels == c].mean(axis=0))
        if len(centroids) < 2:
            return ''
        Z = linkage(np.vstack(centroids), method='average', metric='euclidean')
        tree, _ = to_tree(Z, rd=True)
        return self._to_newick(tree, cluster_ids)

    def _to_newick(self, node, cluster_ids, newick='') -> str:  # type: ignore
        # Recursively build Newick from scipy linkage tree
        if node.is_leaf():
            return f"{cluster_ids[node.id]}:0.0" + (newick if newick else ';')
        else:
            left = self._to_newick(node.get_left(), cluster_ids, '')
            right = self._to_newick(node.get_right(), cluster_ids, '')
            branch = f"({left},{right})" + (newick if newick else ';')
            return branch

"""Simple phylogenetic validation utilities.

This module provides a lightweight, dependency-minimal phylogenetic validation
approach consistent with the documentation promises. Rather than invoking
external tools (IQ-TREE / RAxML) it constructs an approximate tree using:

1. K-mer (k=6 by default) frequency vectors for each sequence (restricted length)
2. Cosine distance between vectors (robust for sparse high‑dimensional counts)
3. Hierarchical clustering (average linkage) via SciPy to obtain a dendrogram
4. Conversion of the linkage matrix to a Newick string for downstream use

Outputs:
  - Newick tree string (saved to results/reports/wnv_tree.newick)
  - Distance matrix (NPZ)
  - Optional lineage concordance statistics if lineage labels available

This serves as an internal sanity check and reproducible placeholder until a
full maximum likelihood workflow is integrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
from collections import Counter

from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from src.utils.logger import setup_logger


@dataclass
class PhylogenyResult:
    newick: str
    distance_matrix_path: Path
    tree_path: Path
    lineage_concordance: Optional[Dict[str, float]] = None


class PhylogeneticValidator:
    """Approximate phylogenetic validator.

    Parameters
    ----------
    k : int
        K‑mer size for frequency vectors.
    max_sequences : int
        Maximum number of sequences to include (random subsample for speed).
    random_state : int
        RNG seed for reproducibility.
    logger : logging.Logger | None
        Optional preconfigured logger.
    """

    def __init__(
        self,
        k: int = 6,
        max_sequences: int = 300,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.k = k
        self.max_sequences = max_sequences
        self.random_state = random_state
        self.logger = logger or setup_logger("phylogeny")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def build_tree(
        self,
        sequences: List[str],
        metadata: List[Dict],
        output_dir: str | Path,
        lineage_key: str = "lineage",
    ) -> PhylogenyResult:
        if len(sequences) == 0:
            raise ValueError("No sequences supplied for phylogenetic validation")

        rng = np.random.default_rng(self.random_state)
        indices = np.arange(len(sequences))
        if len(indices) > self.max_sequences:
            indices = rng.choice(indices, size=self.max_sequences, replace=False)
            self.logger.info(
                f"Subsampled {len(indices)} / {len(sequences)} sequences for tree (max_sequences={self.max_sequences})"
            )

        sub_sequences = [sequences[i] for i in indices]
        sub_metadata = [metadata[i] for i in indices]

        self.logger.info(
            f"Computing k-mer (k={self.k}) frequency vectors for {len(sub_sequences)} sequences"
        )
        kmer_vectors, kmer_list = self._compute_kmer_matrix(sub_sequences)

        # Distance matrix (cosine distance)
        norms = np.linalg.norm(kmer_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = kmer_vectors / norms
        sim = unit @ unit.T
        # Ensure numerical stability
        sim = np.clip(sim, -1, 1)
        dist = 1 - sim
        np.fill_diagonal(dist, 0.0)

        # Condensed distance vector for linkage
        condensed = squareform(dist, checks=False)
        self.logger.info("Performing hierarchical clustering (average linkage)")
        Z = linkage(condensed, method="average")

        newick = self._linkage_to_newick(Z, [m.get("sequence_id", f"seq_{i}") for i, m in enumerate(sub_metadata)])

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tree_path = output_dir / "wnv_tree.newick"
        dist_path = output_dir / "phylogeny_distance_matrix.npz"

        tree_path.write_text(newick)
        np.savez_compressed(dist_path, distance_matrix=dist, indices=indices, k=kmer_list)
        self.logger.info(f"Saved Newick tree: {tree_path}")
        self.logger.info(f"Saved distance matrix: {dist_path}")

        lineage_concordance = self._compute_lineage_concordance(sub_metadata, lineage_key)

        return PhylogenyResult(
            newick=newick,
            distance_matrix_path=dist_path,
            tree_path=tree_path,
            lineage_concordance=lineage_concordance,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_kmer_matrix(self, sequences: List[str]) -> Tuple[np.ndarray, List[str]]:
        bases = ["A", "C", "G", "T"]
        # Generate lexicographically to maintain consistent ordering
        from itertools import product

        kmer_list = ["".join(p) for p in product(bases, repeat=self.k)]
        idx_map = {k: i for i, k in enumerate(kmer_list)}
        mat = np.zeros((len(sequences), len(kmer_list)), dtype=np.float32)

        for row, seq in enumerate(sequences):
            seq = seq.upper()
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i : i + self.k]
                if set(kmer) <= set(bases):
                    mat[row, idx_map[kmer]] += 1
            total = mat[row].sum()
            if total > 0:
                mat[row] /= total
        return mat, kmer_list

    def _linkage_to_newick(self, Z, labels: List[str]) -> str:
        # Adapted minimal conversion from linkage matrix to Newick
        tree, _ = to_tree(Z, rd=True)

        def build(node) -> str:
            if node.is_leaf():
                return labels[node.id]
            left = build(node.get_left())
            right = build(node.get_right())
            # Branch lengths: use node.dist/2 for children (average linkage semantics)
            return f"({left}:{node.dist/2:.5f},{right}:{node.dist/2:.5f})"

        return build(tree) + ";"

    def _compute_lineage_concordance(self, metadata: List[Dict], lineage_key: str) -> Optional[Dict[str, float]]:
        lineages = [m.get(lineage_key) for m in metadata if m.get(lineage_key)]
        if not lineages:
            return None
        counts = Counter(lineages)
        total = sum(counts.values())
        return {lin: c / total for lin, c in counts.items()}


__all__ = ["PhylogeneticValidator", "PhylogenyResult"]

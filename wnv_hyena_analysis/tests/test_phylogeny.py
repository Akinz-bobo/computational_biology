"""Tests for the approximate phylogenetic validator."""

from src.validation.phylogeny import PhylogeneticValidator


def test_phylogeny_basic():
    seqs = [
        "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTT",
        "GTACGTACGTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "GTACGTACGTAGCTAGCTAGCTAGCTAGCTAGCTT",
    ]
    meta = [{"sequence_id": f"s{i}", "lineage": "L1" if i < 2 else "L2"} for i in range(len(seqs))]
    validator = PhylogeneticValidator(k=3, max_sequences=10)
    result = validator.build_tree(seqs, meta, "results/reports/test_phylo")
    assert result.newick.endswith(";")
    assert result.lineage_concordance is not None

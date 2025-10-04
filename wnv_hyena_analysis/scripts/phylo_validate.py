#!/usr/bin/env python3
"""Run approximate phylogenetic validation for WNV sequences.

Usage:
  python scripts/phylo_validate.py --config config/config.yaml --n 250
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.analysis.data_loader import WNVDataLoader
from src.validation.phylogeny import PhylogeneticValidator
from src.utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Approximate phylogenetic validation (k-mer + clustering)")
    p.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    p.add_argument("--n", type=int, default=300, help="Max sequences to include")
    p.add_argument("--k", type=int, default=6, help="K-mer size (default 6)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(args.config)
    logger = setup_logger("phylo_run")

    fasta = cfg.get("data.raw_fasta")
    if not Path(fasta).exists():
        raise SystemExit(f"FASTA not found: {fasta}")

    loader = WNVDataLoader(cfg._config, logger)
    sequences, metadata = loader.load_fasta(fasta)

    validator = PhylogeneticValidator(k=args.k, max_sequences=args.n, logger=logger)
    out_dir = Path(cfg.get("results.reports_dir")) / "phylogeny"
    result = validator.build_tree(sequences, metadata, out_dir)

    logger.info("Phylogenetic validation complete")
    if result.lineage_concordance:
        logger.info(f"Lineage proportions: {result.lineage_concordance}")
    print(f"Newick tree saved to: {result.tree_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

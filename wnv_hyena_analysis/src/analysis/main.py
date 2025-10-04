"""Entry point for `wnv-analyze` console script.

This mirrors the functionality of `scripts/run_analysis.py` (kept for local
execution) so that installed packages do not rely on the non-packaged `scripts`
directory. Only a streamlined subset of arguments is supported here; for full
options invoke the script directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from datetime import datetime
import json
import numpy as np

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.analysis.data_loader import WNVDataLoader
from src.analysis.feature_extractor import FeatureExtractor
from src.models.classifier import WNVClassifier
from src.visualization.plotter import WNVPlotter
from src.validation.phylogeny import PhylogeneticValidator


def _parse_args():
	p = argparse.ArgumentParser(
		description="WNV HyenaDNA classification (packaged entry point)",
	)
	p.add_argument("--config", default="config/config.yaml")
	p.add_argument("--n-samples", type=int, default=100, help="Sample size (use --full for all)")
	p.add_argument("--full", action="store_true", help="Use all sequences")
	p.add_argument("--traditional-only", action="store_true")
	p.add_argument("--hyenadna-only", action="store_true")
	p.add_argument("--phylo", action="store_true", help="Run approximate phylogenetic validation")
	return p.parse_args()


def main():  # pragma: no cover - high level orchestration
	args = _parse_args()
	cfg = Config(args.config)
	cfg.create_directories()
	log_file = Path("logs") / f"wnv_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
	log_file.parent.mkdir(exist_ok=True)
	logger = setup_logger("wnv_entry", log_file=str(log_file))

	fasta = cfg.get("data.raw_fasta")
	if not Path(fasta).exists():
		raise SystemExit(f"FASTA file not found: {fasta}")

	# Data load
	sample_size = None if args.full else args.n_samples
	loader = WNVDataLoader(cfg._config, logger)
	sequences, metadata = loader.load_fasta(fasta, sample_size=sample_size)
	df = loader.create_dataframe()

	# Features
	if args.traditional_only:
		cfg.set("features.deep_learning.enabled", False)
	if args.hyenadna_only:
		cfg.set("features.traditional.enabled", False)
	extractor = FeatureExtractor(cfg._config, logger)
	traditional, deep, names = extractor.extract_all_features(sequences)
	if args.traditional_only:
		feats = traditional
		active_names = names[: traditional.shape[1]]
	elif args.hyenadna_only:
		feats = deep
		active_names = names[traditional.shape[1] :]
	else:
		feats = np.hstack([traditional, deep])
		active_names = names

	# Classification
	target = cfg.get("classification.target_column", "country")
	if target not in df.columns:
		target = "country"
	mask = (df[target] != "Unknown") & df[target].notna()
	X = feats[mask]
	y = df[target][mask].values
	classifier = WNVClassifier(cfg._config, logger)
	results = classifier.train_and_evaluate(X, y, active_names)

	# Visualizations
	plotter = WNVPlotter(cfg._config, logger)
	plotter.generate_all_plots(df, feats, active_names, results)

	# Save minimal outputs
	out_reports = Path(cfg.get("results.reports_dir"))
	out_reports.mkdir(parents=True, exist_ok=True)
	(out_reports / "classification_results.json").write_text(json.dumps(results, default=str, indent=2))

	if args.phylo:
		validator = PhylogeneticValidator(logger=logger)
		validator.build_tree(sequences, metadata, out_reports / "phylogeny")

	print("Analysis complete. Best model:", results.get("best_model"))


if __name__ == "__main__":  # pragma: no cover
	main()


#!/usr/bin/env python3
"""
Main analysis script for WNV HyenaDNA Classification
Publication-ready bioinformatics pipeline
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.analysis.data_loader import WNVDataLoader
from src.analysis.feature_extractor import FeatureExtractor
from src.models.classifier import WNVClassifier
from src.visualization.plotter import WNVPlotter


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="WNV HyenaDNA Classification Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --quick --n-samples 100
  python run_analysis.py --full --reproducible
  python run_analysis.py --traditional-only --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run quick analysis with sample data"
    )
    
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Run full analysis with all sequences"
    )
    
    parser.add_argument(
        "--n-samples", 
        type=int,
        help="Number of sequences to sample (overrides config)"
    )
    
    parser.add_argument(
        "--traditional-only", 
        action="store_true",
        help="Use only traditional bioinformatics features"
    )
    
    parser.add_argument(
        "--hyenadna-only", 
        action="store_true",
        help="Use only HyenaDNA deep learning features"
    )
    
    parser.add_argument(
        "--reproducible", 
        action="store_true",
        help="Enable full reproducibility mode"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Output directory (overrides config)"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--phylo",
        action="store_true",
        help="Run approximate phylogenetic validation (k-mer clustering)"
    )
    parser.add_argument(
        "--infer-lineages",
        action="store_true",
        help="Infer putative lineages via unsupervised clustering and association tests"
    )
    
    return parser.parse_args()


def main():
    """Main analysis pipeline"""
    args = parse_arguments()
    
    # Load configuration
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Create directories
    config.create_directories()
    
    # Setup logging
    log_file = Path("logs") / f"wnv_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(exist_ok=True)
    
    logger = setup_logger(
        "wnv_analysis", 
        level=args.log_level,
        log_file=str(log_file),
        console=True
    )
    
    logger.info("="*80)
    logger.info("WNV HyenaDNA Classification Analysis Pipeline")
    logger.info("Publication-ready bioinformatics analysis")
    logger.info("="*80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Log file: {log_file}")
    
    try:
        # Determine sample size
        sample_size = None
        if args.quick and not args.full:
            sample_size = args.n_samples or 100
            logger.info(f"Running quick analysis with {sample_size} samples")
        elif args.n_samples:
            sample_size = args.n_samples
            logger.info(f"Using {sample_size} samples as specified")
        elif args.full:
            logger.info("Running full analysis with all sequences")
        else:
            logger.info("Using default configuration settings")
        
        # Step 1: Load data
        logger.info("Step 1: Loading WNV genome data")
        data_loader = WNVDataLoader(config._config, logger)
        
        fasta_path = config.get('data.raw_fasta')
        if not Path(fasta_path).exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
        sequences, metadata = data_loader.load_fasta(fasta_path, sample_size)
        df = data_loader.create_dataframe()
        
        logger.info(f"Loaded {len(sequences)} sequences")
        logger.info(f"Dataset info: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Step 2: Feature extraction
        logger.info("Step 2: Feature extraction")
        
        # Configure feature extraction based on arguments
        if args.traditional_only:
            config.set('features.deep_learning.enabled', False)
            logger.info("Using traditional features only")
        elif args.hyenadna_only:
            config.set('features.traditional.enabled', False)
            logger.info("Using HyenaDNA features only")
        
        feature_extractor = FeatureExtractor(config._config, logger)
        traditional_features, deep_features, feature_names = feature_extractor.extract_all_features(sequences)
        
        # Combine features
        if args.traditional_only:
            combined_features = traditional_features
            active_feature_names = feature_names[:traditional_features.shape[1]]
        elif args.hyenadna_only:
            combined_features = deep_features
            active_feature_names = feature_names[traditional_features.shape[1]:]
        else:
            combined_features = np.hstack([traditional_features, deep_features])
            active_feature_names = feature_names
        
        logger.info(f"Final feature matrix shape: {combined_features.shape}")
        
        # Step 3: Classification analysis
        logger.info("Step 3: Classification analysis")
        
        # Determine target variable
        target_col = config.get('classification.target_column', 'country')
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found, using 'country'")
            target_col = 'country'
        
        # Remove samples with unknown/missing target values
        if target_col == 'lineage':
            valid_mask = df[target_col].notna() & (df[target_col] != 'Unknown')
        else:
            valid_mask = df[target_col] != 'Unknown'
        
        if not valid_mask.any():
            logger.warning(f"No valid samples for target '{target_col}', switching to 'country'")
            target_col = 'country'
            valid_mask = df[target_col] != 'Unknown'
            
        if not valid_mask.any():
            raise ValueError(f"No valid samples found for classification")
        
        X = combined_features[valid_mask]
        y = df[target_col][valid_mask].values
        
        logger.info(f"Classification setup: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target variable: {target_col}")
        logger.info(f"Classes: {np.unique(y)}")
        
        # Initialize classifier
        classifier = WNVClassifier(config._config, logger)
        
        # Train and evaluate models
        results = classifier.train_and_evaluate(X, y, active_feature_names)
        
        # Step 4: Generate visualizations
        logger.info("Step 4: Generating visualizations")
        
        plotter = WNVPlotter(config._config, logger)
        
        # Generate comprehensive plots
        plot_results = plotter.generate_all_plots(
            df=df,
            features=combined_features,
            feature_names=active_feature_names,
            classification_results=results
        )
        
        # Step 5: Save results
        logger.info("Step 5: Saving results")
        
        # Save processed data
        processed_path = Path(config.get('data.processed_dir')) / 'wnv_processed'
        data_loader.save_processed_data(df, processed_path)
        
        # Save features
        features_path = Path(config.get('data.processed_dir')) / 'wnv_features.npz'
        np.savez_compressed(
            features_path,
            traditional_features=traditional_features,
            deep_features=deep_features,
            combined_features=combined_features,
            feature_names=active_feature_names
        )
        logger.info(f"Saved features to: {features_path}")
        
        # Save classification results
        results_path = Path(config.get('results.reports_dir')) / 'classification_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in value.items()}
            else:
                json_results[key] = value.tolist() if hasattr(value, 'tolist') else value
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        logger.info(f"Saved classification results to: {results_path}")
        
        # Generate summary report
        summary = generate_summary_report(df, results, config._config)
        summary_path = Path(config.get('results.reports_dir')) / 'analysis_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        logger.info(f"Saved summary report to: {summary_path}")

        # Optional phylogenetic validation
        if args.phylo:
            try:
                from src.validation.phylogeny import PhylogeneticValidator
                logger.info("Running phylogenetic validation (approximate)")
                validator = PhylogeneticValidator()
                phylo_dir = Path(config.get('results.reports_dir')) / 'phylogeny'
                validator.build_tree(sequences, metadata, phylo_dir)
            except Exception as e:
                logger.error(f"Phylogenetic validation failed: {e}")

        # Optional lineage inference
        if args.infer_lineages:
            try:
                from src.analysis.lineage_inference import LineageInferer
                logger.info("Running lineage inference (unsupervised clustering)")
                inferer = LineageInferer(config._config, logger)
                lineage_results = inferer.infer(combined_features, df)
                lineage_dir = Path(config.get('results.reports_dir'))
                lineage_path = lineage_dir / 'lineage_inference.json'
                # Convert numpy types recursively for JSON safety
                def _convert(obj):
                    if isinstance(obj, dict):
                        return {k: _convert(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_convert(v) for v in obj]
                    if hasattr(obj, 'tolist'):
                        try:
                            return obj.tolist()
                        except Exception:
                            return str(obj)
                    if isinstance(obj, (np.floating, np.integer)):
                        return obj.item()
                    return obj
                lineage_results_clean = _convert(lineage_results)
                with open(lineage_path, 'w') as f:
                    json.dump(lineage_results_clean, f, indent=2)
                logger.info(f"Saved lineage inference results to: {lineage_path}")
            except Exception as e:
                logger.error(f"Lineage inference failed: {e}")
        
        logger.info("="*80)
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved in: {config.get('results.reports_dir')}")
        logger.info(f"Figures saved in: {config.get('visualization.figures_dir')}")
        logger.info("="*80)
        
        # Print key results
        print("\n" + "="*60)
        print("WNV HYENADNA CLASSIFICATION - ANALYSIS COMPLETE")
        print("="*60)
        print(f"Sequences analyzed: {len(sequences):,}")
        print(f"Features extracted: {combined_features.shape[1]:,}")
        print(f"Classification target: {target_col}")
        
        if 'best_model' in results:
            best_model = results['best_model']
            best_score = results.get('best_score', 'N/A')
            print(f"Best model: {best_model}")
            print(f"Best score: {best_score:.4f}" if isinstance(best_score, float) else f"Best score: {best_score}")
        
        print(f"\nDetailed results: {results_path}")
        print(f"Summary report: {summary_path}")
        print(f"Log file: {log_file}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


def generate_summary_report(df: pd.DataFrame, results: dict, config: dict) -> str:
    """Generate a comprehensive summary report"""
    
    report = []
    report.append("WNV HYENADNA CLASSIFICATION - ANALYSIS SUMMARY")
    report.append("=" * 80)
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Configuration: {config.get('project', {}).get('name', 'WNV Analysis')}")
    report.append("")
    
    # Dataset summary
    report.append("DATASET SUMMARY")
    report.append("-" * 40)
    report.append(f"Total sequences: {len(df):,}")
    report.append(f"Average sequence length: {df['length'].mean():.0f} bp")
    report.append(f"Sequence length range: {df['length'].min()}-{df['length'].max()} bp")
    report.append(f"Average GC content: {df['gc_content'].mean():.2f}%")
    report.append("")
    
    # Geographic distribution
    if 'country' in df.columns:
        country_counts = df['country'].value_counts()
        report.append("Geographic Distribution:")
        for country, count in country_counts.head(10).items():
            report.append(f"  {country}: {count:,} sequences")
        if len(country_counts) > 10:
            report.append(f"  ... and {len(country_counts) - 10} more countries")
        report.append("")
    
    # Temporal distribution
    if 'year' in df.columns and df['year'].notna().any():
        year_range = f"{df['year'].min():.0f}-{df['year'].max():.0f}"
        report.append(f"Temporal range: {year_range}")
        report.append("")
    
    # Classification results
    if results:
        report.append("CLASSIFICATION RESULTS")
        report.append("-" * 40)
        
        if 'best_model' in results:
            report.append(f"Best performing model: {results['best_model']}")
        
        if 'best_score' in results:
            score = results['best_score']
            if isinstance(score, (int, float)):
                report.append(f"Best cross-validation score: {score:.4f}")
        
        if 'model_scores' in results:
            report.append("\nAll model performances:")
            for model, score in results['model_scores'].items():
                if isinstance(score, (int, float)):
                    report.append(f"  {model}: {score:.4f}")
        
        report.append("")
    
    # Feature importance (if available)
    if 'feature_importance' in results:
        report.append("TOP FEATURE IMPORTANCE")
        report.append("-" * 40)
        importance = results['feature_importance']
        if isinstance(importance, dict):
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, score in sorted_features:
                report.append(f"  {feature}: {score:.4f}")
        report.append("")
    
    report.append("REPRODUCIBILITY")
    report.append("-" * 40)
    report.append(f"Random seed: {config.get('reproducibility', {}).get('random_seed', 'Not set')}")
    report.append(f"Configuration file: {config.get('project', {}).get('name', 'config.yaml')}")
    report.append("")
    
    report.append("Analysis completed successfully!")
    
    return "\n".join(report)


if __name__ == "__main__":
    main()
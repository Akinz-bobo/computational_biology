"""
Publication-quality visualization for WNV genome analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Optional imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from src.utils.logger import setup_logger


class WNVPlotter:
    """Publication-quality plotting for WNV analysis results"""
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or setup_logger("wnv_plotter")
        
        # Configuration
        self.figures_dir = Path(config.get('visualization', {}).get('figures_dir', 'results/figures/'))
        self.dpi = config.get('visualization', {}).get('dpi', 300)
        self.formats = config.get('visualization', {}).get('format', ['png', 'pdf'])
        
        # Create output directory
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib for publication quality
        self._setup_plotting_style()
        
        self.logger.info(f"Initialized plotter with output directory: {self.figures_dir}")
    
    def _setup_plotting_style(self):
        """Configure matplotlib and seaborn for publication-quality plots"""
        # Publication style settings
        plt.rcParams.update({
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'lines.linewidth': 2,
            'patch.linewidth': 0.5,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': False
        })
        
        # Set seaborn style
        sns.set_palette("husl")
        sns.set_style("whitegrid")
    
    def save_figure(self, fig, filename: str, tight_layout: bool = True):
        """Save figure in multiple formats"""
        if tight_layout:
            fig.tight_layout()
        
        for fmt in self.formats:
            filepath = self.figures_dir / f"{filename}.{fmt}"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        self.logger.info(f"Saved figure: {filename}")
        plt.close(fig)
    
    def plot_sequence_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Plot sequence length and quality statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('West Nile Virus Genome Statistics', fontsize=16, fontweight='bold')
        
        # Sequence length distribution
        axes[0, 0].hist(df['length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Sequence Length (bp)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Sequence Length Distribution')
        axes[0, 0].axvline(df['length'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["length"].mean():.0f} bp')
        axes[0, 0].legend()
        
        # GC content distribution
        axes[0, 1].hist(df['gc_content'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('GC Content (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('GC Content Distribution')
        axes[0, 1].axvline(df['gc_content'].mean(), color='red', linestyle='--',
                          label=f'Mean: {df["gc_content"].mean():.1f}%')
        axes[0, 1].legend()
        
        # N content (quality indicator)
        axes[1, 0].hist(df['n_content'], bins=30, alpha=0.7, color='coral', edgecolor='black')
        axes[1, 0].set_xlabel('N Content (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Sequence Quality (N Content)')
        
        # Sequence length vs GC content
        scatter = axes[1, 1].scatter(df['length'], df['gc_content'], alpha=0.6, 
                                   c=df['n_content'], cmap='viridis', s=20)
        axes[1, 1].set_xlabel('Sequence Length (bp)')
        axes[1, 1].set_ylabel('GC Content (%)')
        axes[1, 1].set_title('Length vs GC Content')
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('N Content (%)')
        
        self.save_figure(fig, 'sequence_statistics')
        
        return {
            'mean_length': df['length'].mean(),
            'std_length': df['length'].std(),
            'mean_gc': df['gc_content'].mean(),
            'std_gc': df['gc_content'].std()
        }
    
    def plot_geographic_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Plot geographic distribution of sequences"""
        # Country distribution
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        fig.suptitle('Geographic Distribution of WNV Sequences', fontsize=16, fontweight='bold')
        
        # Top countries bar plot
        country_counts = df['country'].value_counts().head(15)
        axes[0].bar(range(len(country_counts)), country_counts.values, 
                   color=sns.color_palette("viridis", len(country_counts)))
        axes[0].set_xticks(range(len(country_counts)))
        axes[0].set_xticklabels(country_counts.index, rotation=45, ha='right')
        axes[0].set_ylabel('Number of Sequences')
        axes[0].set_title('Top 15 Countries by Sequence Count')
        
        # Add value labels on bars
        for i, v in enumerate(country_counts.values):
            axes[0].text(i, v + max(country_counts.values) * 0.01, str(v), 
                        ha='center', va='bottom')
        
        # Continental distribution pie chart
        continent_counts = df['continent'].value_counts()
        colors = sns.color_palette("Set2", len(continent_counts))
        wedges, texts, autotexts = axes[1].pie(continent_counts.values, labels=continent_counts.index,
                                              autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1].set_title('Continental Distribution')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        self.save_figure(fig, 'geographic_distribution')
        
        return {
            'top_countries': country_counts.to_dict(),
            'continental_distribution': continent_counts.to_dict()
        }
    
    def plot_temporal_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Plot temporal distribution of sequences"""
        # Filter out missing years
        df_temporal = df[df['year'].notna()].copy()
        
        if len(df_temporal) == 0:
            self.logger.warning("No temporal data available for plotting")
            return {}
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Temporal Distribution of WNV Sequences', fontsize=16, fontweight='bold')
        
        # Year distribution histogram
        year_counts = df_temporal['year'].value_counts().sort_index()
        axes[0].plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=6)
        axes[0].fill_between(year_counts.index, year_counts.values, alpha=0.3)
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of Sequences')
        axes[0].set_title('Sequence Collection Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        cumsum = year_counts.cumsum()
        axes[1].plot(cumsum.index, cumsum.values, marker='s', linewidth=2, 
                    markersize=4, color='red')
        axes[1].fill_between(cumsum.index, cumsum.values, alpha=0.3, color='red')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Cumulative Sequence Count')
        axes[1].set_title('Cumulative Sequence Collection')
        axes[1].grid(True, alpha=0.3)
        
        self.save_figure(fig, 'temporal_distribution')
        
        return {
            'year_range': (int(df_temporal['year'].min()), int(df_temporal['year'].max())),
            'total_with_year': len(df_temporal),
            'peak_year': int(year_counts.idxmax()),
            'peak_count': int(year_counts.max())
        }
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              top_n: int = 20) -> Dict[str, Any]:
        """Plot feature importance from best model"""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color=sns.color_palette("viridis", len(features)))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Most Important Features')
        
        # Add value labels on bars
        for i, (feature, importance) in enumerate(top_features):
            ax.text(importance + max(importances) * 0.01, i, f'{importance:.3f}', 
                   va='center', fontsize=9)
        
        self.save_figure(fig, 'feature_importance')
        
        return {
            'top_features': dict(top_features),
            'total_features': len(feature_importance)
        }
    
    def plot_classification_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Plot classification performance results"""
        model_results = results.get('model_results', {})
        
        if not model_results:
            self.logger.warning("No classification results to plot")
            return {}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Classification Model Performance', fontsize=16, fontweight='bold')
        
        # Model comparison - CV scores
        models = list(model_results.keys())
        cv_means = [model_results[model]['mean_cv_score'] for model in models]
        cv_stds = [model_results[model]['std_cv_score'] for model in models]
        
        x_pos = np.arange(len(models))
        bars = axes[0, 0].bar(x_pos, cv_means, yerr=cv_stds, capsize=5, 
                             color=sns.color_palette("Set2", len(models)), alpha=0.8)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Cross-Validation Score')
        axes[0, 0].set_title('Model Comparison (CV Scores)')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            axes[0, 0].text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom')
        
        # Test accuracy comparison
        test_accuracies = [model_results[model]['test_accuracy'] for model in models]
        axes[0, 1].bar(x_pos, test_accuracies, color=sns.color_palette("Set1", len(models)), alpha=0.8)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].set_title('Test Set Performance')
        
        for i, acc in enumerate(test_accuracies):
            axes[0, 1].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        # Confusion matrix for best model
        best_model = results.get('best_model')
        if best_model and best_model in model_results:
            cm = np.array(model_results[best_model]['confusion_matrix'])
            labels = list(results.get('label_mapping', {}).keys())
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels, ax=axes[1, 0])
            axes[1, 0].set_title(f'Confusion Matrix - {best_model}')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
        
        # F1 scores comparison
        f1_scores = [model_results[model]['test_f1_macro'] for model in models]
        axes[1, 1].bar(x_pos, f1_scores, color=sns.color_palette("Pastel1", len(models)), alpha=0.8)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].set_ylabel('F1-Score (Macro)')
        axes[1, 1].set_title('F1-Score Comparison')
        
        for i, f1 in enumerate(f1_scores):
            axes[1, 1].text(i, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
        
        self.save_figure(fig, 'classification_results')
        
        return {
            'best_model': best_model,
            'best_cv_score': max(cv_means) if cv_means else None,
            'best_test_accuracy': max(test_accuracies) if test_accuracies else None,
            'model_count': len(models)
        }
    
    def plot_dimensionality_reduction(self, features: np.ndarray, labels: np.ndarray, 
                                    method: str = 'PCA') -> Dict[str, Any]:
        """Plot dimensionality reduction visualization"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available, skipping dimensionality reduction plot")
            return {}
        
        # Limit samples for visualization efficiency
        max_samples = 1000
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features_sample = features[indices]
            labels_sample = labels[indices]
        else:
            features_sample = features
            labels_sample = labels
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Dimensionality Reduction Visualization', fontsize=16, fontweight='bold')
        
        # PCA
        if method in ['PCA', 'all']:
            pca = PCA(n_components=2, random_state=42)
            features_pca = pca.fit_transform(features_sample)
            
            scatter = axes[0].scatter(features_pca[:, 0], features_pca[:, 1], 
                                    c=[hash(label) % 20 for label in labels_sample], 
                                    cmap='tab20', alpha=0.7, s=30)
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            axes[0].set_title('PCA Visualization')
            
            # Add legend for unique labels (limit to prevent crowding)
            unique_labels = np.unique(labels_sample)
            if len(unique_labels) <= 10:
                handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=plt.cm.tab20(hash(label) % 20 / 20), 
                                    markersize=8, label=label) for label in unique_labels]
                axes[0].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # t-SNE
        if method in ['TSNE', 'all']:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_sample)//4))
            features_tsne = tsne.fit_transform(features_sample)
            
            scatter = axes[1].scatter(features_tsne[:, 0], features_tsne[:, 1],
                                    c=[hash(label) % 20 for label in labels_sample],
                                    cmap='tab20', alpha=0.7, s=30)
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
            axes[1].set_title('t-SNE Visualization')
        
        self.save_figure(fig, f'dimensionality_reduction_{method.lower()}')
        
        return {
            'method': method,
            'samples_visualized': len(features_sample),
            'unique_labels': len(np.unique(labels_sample))
        }
    
    def generate_all_plots(self, df: pd.DataFrame, features: np.ndarray, 
                          feature_names: List[str], classification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all publication-quality plots"""
        self.logger.info("Generating comprehensive visualization suite")
        
        plot_results = {}
        
        try:
            # Sequence statistics
            plot_results['sequence_stats'] = self.plot_sequence_statistics(df)
            
            # Geographic distribution
            plot_results['geographic'] = self.plot_geographic_distribution(df)
            
            # Temporal distribution
            plot_results['temporal'] = self.plot_temporal_distribution(df)
            
            # Feature importance (if available)
            if 'feature_importance' in classification_results:
                feature_importance = classification_results['feature_importance']
                plot_results['feature_importance'] = self.plot_feature_importance(feature_importance)
            
            # Classification results
            plot_results['classification'] = self.plot_classification_results(classification_results)
            
            # Dimensionality reduction
            target_col = 'country'  # Default target
            if target_col in df.columns:
                valid_mask = df[target_col] != 'Unknown'
                if valid_mask.any():
                    labels = df[target_col][valid_mask].values
                    features_valid = features[valid_mask]
                    plot_results['dimensionality_reduction'] = self.plot_dimensionality_reduction(
                        features_valid, labels, method='PCA'
                    )
            
            self.logger.info("All plots generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
            raise
        
        return plot_results

    # ---------------- Lineage inference visualizations ----------------
    def plot_lineage_distribution(self, lineage_results: Dict[str, Any]):
        if not lineage_results or lineage_results.get('skipped'):
            self.logger.warning("No lineage results to plot")
            return {}
        labels = lineage_results.get('lineage_labels')
        if labels is None:
            return {}
        labels = np.array(labels)
        unique, counts = np.unique(labels, return_counts=True)
        fig, ax = plt.subplots(figsize=(10,6))
        order = np.argsort(-counts)
        ax.bar(range(len(order)), counts[order], color=sns.color_palette('tab20', len(order)))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(unique[order], rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title('Inferred Lineage Size Distribution')
        for i, c in enumerate(counts[order]):
            ax.text(i, c + max(counts)*0.01, str(c), ha='center', va='bottom')
        self.save_figure(fig, 'lineage_distribution')
        return {"lineage_sizes": dict(zip(unique, counts))}

    def plot_lineage_associations(self, lineage_results: Dict[str, Any]):
        if not lineage_results or 'associations' not in lineage_results:
            return {}
        assoc = lineage_results['associations']
        output = {}
        for name, stats in assoc.items():
            if not stats.get('available'):
                continue
            table = stats.get('table')
            if not table:
                continue
            df_table = pd.DataFrame(table).T  # lineage x category
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(df_table, annot=False, cmap='viridis', ax=ax)
            ax.set_title(f'Lineage vs {name.title()} Association (Chi2 p={stats.get("p_value"):.2e}, Cramer V={stats.get("cramers_v"):.2f})')
            ax.set_xlabel(name.title())
            ax.set_ylabel('Lineage')
            self.save_figure(fig, f'lineage_{name}_association')
            output[name] = {"p_value": stats.get('p_value'), "cramers_v": stats.get('cramers_v')}
        return output
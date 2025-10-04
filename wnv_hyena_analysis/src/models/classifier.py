"""
Machine learning classifiers for WNV genome analysis
Publication-ready classification with comprehensive evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, f1_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.utils.logger import setup_logger


class WNVClassifier:
    """
    Comprehensive classifier for WNV genome classification
    Supports multiple algorithms with hyperparameter tuning and evaluation
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or setup_logger("wnv_classifier")
        
        # Configuration
        self.test_size = config.get('classification', {}).get('test_size', 0.2)
        self.validation_size = config.get('classification', {}).get('validation_size', 0.2)
        self.random_state = config.get('classification', {}).get('random_state', 42)
        self.cv_folds = config.get('classification', {}).get('hyperparameter_tuning', {}).get('cv_folds', 5)
        self.scoring = config.get('classification', {}).get('hyperparameter_tuning', {}).get('scoring', 'f1_macro')
        
        # Models to evaluate
        self.model_names = config.get('classification', {}).get('models', [
            'RandomForest', 'GradientBoosting', 'SVM', 'LogisticRegression'
        ])
        
        if XGBOOST_AVAILABLE and 'XGBoost' in self.model_names:
            self.logger.info("XGBoost available and enabled")
        elif 'XGBoost' in self.model_names:
            self.logger.warning("XGBoost requested but not available, skipping")
            self.model_names = [name for name in self.model_names if name != 'XGBoost']
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.best_score = -1
        
        # Results storage
        self.results = {}
    
    def get_model(self, model_name: str) -> Any:
        """Get model instance by name"""
        models = {
            'RandomForest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'SVM': SVC(
                random_state=self.random_state,
                probability=True  # For ROC curves
            ),
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=-1
            ),
            'KNN': KNeighborsClassifier(
                n_jobs=-1
            )
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='mlogloss',
                n_jobs=-1
            )
        
        return models.get(model_name)
    
    def get_hyperparameter_space(self, model_name: str) -> Dict:
        """Get hyperparameter search space for each model"""
        spaces = {
            'RandomForest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'solver': ['lbfgs', 'liblinear', 'saga']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
        
        if XGBOOST_AVAILABLE:
            spaces['XGBoost'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        return spaces.get(model_name, {})
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training: scaling and train/test split"""
        self.logger.info("Preparing data for training")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.logger.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Train/test split
        stratify = y_encoded if self.config.get('classification', {}).get('stratify', True) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"Training set: {X_train_scaled.shape[0]} samples")
        self.logger.info(f"Test set: {X_test_scaled.shape[0]} samples")
        self.logger.info(f"Features: {X_train_scaled.shape[1]}")
        self.logger.info(f"Classes: {len(np.unique(y_encoded))}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Train and evaluate all configured models
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features for interpretation
            
        Returns:
            Dictionary containing all evaluation results
        """
        self.logger.info("Starting comprehensive model training and evaluation")
        
        # Edge case: single class -> baseline output
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            self.logger.warning("Only one class present in target; skipping model training and returning baseline stats")
            self.label_encoder.fit(y)
            baseline_results = {
                'model_results': {},
                'model_scores': {},
                'best_model': None,
                'best_score': 1.0,
                'label_mapping': {unique_classes[0]: 0},
                'feature_names': feature_names,
                'configuration': {
                    'test_size': self.test_size,
                    'cv_folds': self.cv_folds,
                    'scoring_metric': self.scoring,
                    'random_state': self.random_state
                },
                'note': 'Single-class dataset; no classifier training performed.'
            }
            self.results = baseline_results
            return baseline_results

        # Prepare data normally when multi-class
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Storage for results
        model_results = {}
        model_scores = {}
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Train each model
        for model_name in self.model_names:
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Get base model
                base_model = self.get_model(model_name)
                if base_model is None:
                    self.logger.warning(f"Model {model_name} not available, skipping")
                    continue
                
                # Hyperparameter tuning
                param_space = self.get_hyperparameter_space(model_name)
                
                if param_space:
                    self.logger.info(f"Performing hyperparameter tuning for {model_name}")
                    
                    # Use RandomizedSearchCV for efficiency
                    n_iter = self.config.get('classification', {}).get('hyperparameter_tuning', {}).get('n_iter', 50)
                    
                    search = RandomizedSearchCV(
                        base_model,
                        param_space,
                        n_iter=n_iter,
                        cv=cv,
                        scoring=self.scoring,
                        n_jobs=-1,
                        random_state=self.random_state,
                        verbose=0
                    )
                    
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    
                    self.logger.info(f"Best parameters for {model_name}: {search.best_params_}")
                    self.logger.info(f"Best CV score for {model_name}: {search.best_score_:.4f}")
                    
                else:
                    # Use default parameters
                    best_model = base_model
                    best_model.fit(X_train, y_train)
                
                # Store model
                self.models[model_name] = best_model
                
                # Cross-validation evaluation
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=self.scoring)
                mean_cv_score = cv_scores.mean()
                model_scores[model_name] = mean_cv_score
                
                # Test set evaluation
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
                
                # Calculate metrics
                test_accuracy = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred, average='macro')
                
                # Store results
                model_results[model_name] = {
                    'cv_scores': cv_scores.tolist(),
                    'mean_cv_score': mean_cv_score,
                    'std_cv_score': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'test_f1_macro': test_f1,
                    'predictions': y_pred.tolist(),
                    'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
                    'classification_report': classification_report(
                        y_test, y_pred, 
                        target_names=self.label_encoder.classes_,
                        output_dict=True
                    ),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                # Feature importance (if available)
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    feature_importance = dict(zip(feature_names, importances))
                    model_results[model_name]['feature_importance'] = feature_importance
                
                elif hasattr(best_model, 'coef_'):
                    # For linear models, use coefficient magnitudes
                    if best_model.coef_.ndim == 1:
                        importances = np.abs(best_model.coef_)
                    else:
                        importances = np.mean(np.abs(best_model.coef_), axis=0)
                    
                    feature_importance = dict(zip(feature_names, importances))
                    model_results[model_name]['feature_importance'] = feature_importance
                
                # Track best model
                if mean_cv_score > self.best_score:
                    self.best_score = mean_cv_score
                    self.best_model = model_name
                
                self.logger.info(f"{model_name} - CV Score: {mean_cv_score:.4f} ± {cv_scores.std():.4f}")
                self.logger.info(f"{model_name} - Test Accuracy: {test_accuracy:.4f}")
                self.logger.info(f"{model_name} - Test F1-macro: {test_f1:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Compile final results
        self.results = {
            'model_results': model_results,
            'model_scores': model_scores,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'label_mapping': dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_)))),
            'feature_names': feature_names,
            'test_indices': None,  # Could store if needed
            'configuration': {
                'test_size': self.test_size,
                'cv_folds': self.cv_folds,
                'scoring_metric': self.scoring,
                'random_state': self.random_state
            }
        }
        
        # Add aggregated feature importance
        if self.best_model and self.best_model in model_results:
            best_importance = model_results[self.best_model].get('feature_importance')
            if best_importance:
                self.results['feature_importance'] = best_importance
        
        self.logger.info("Model training and evaluation completed")
        self.logger.info(f"Best model: {self.best_model} (Score: {self.best_score:.4f})")
        
        return self.results
    
    def save_models(self, output_dir: str):
        """Save trained models to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save all models
        for name, model in self.models.items():
            model_path = output_path / f"{name.lower()}_model.pkl"
            joblib.dump(model, model_path)
            self.logger.info(f"Saved {name} model to: {model_path}")
        
        # Save preprocessing objects
        scaler_path = output_path / "scaler.pkl"
        encoder_path = output_path / "label_encoder.pkl"
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        self.logger.info(f"Saved scaler to: {scaler_path}")
        self.logger.info(f"Saved label encoder to: {encoder_path}")
    
    def load_models(self, input_dir: str):
        """Load trained models from disk"""
        input_path = Path(input_dir)
        
        # Load preprocessing objects
        scaler_path = input_path / "scaler.pkl"
        encoder_path = input_path / "label_encoder.pkl"
        
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.logger.info("Loaded scaler")
        
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
            self.logger.info("Loaded label encoder")
        
        # Load models
        for model_name in self.model_names:
            model_path = input_path / f"{model_name.lower()}_model.pkl"
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                self.logger.info(f"Loaded {model_name} model")
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained model
        
        Args:
            X: Feature matrix
            model_name: Name of model to use (uses best model if None)
            
        Returns:
            Tuple of (predictions, prediction_probabilities)
        """
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)
        else:
            probabilities = None
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities
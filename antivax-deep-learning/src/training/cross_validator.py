"""
Cross-validation utilities for model evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Any, Callable
from pathlib import Path
import json


class CrossValidator:
    """
    Perform k-fold cross-validation for neural network models.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """
        Initialize CrossValidator.
        
        Args:
            n_folds: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.cv_results = []
        
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_builder: Callable,
        train_func: Callable,
        evaluate_func: Callable,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Feature array
            y: Target array
            model_builder: Function that builds and returns a new model
            train_func: Function to train model (model, X_train, y_train, X_val, y_val)
            evaluate_func: Function to evaluate model (model, X, y)
            verbose: Verbosity level
            
        Returns:
            Dictionary containing cross-validation results
        """
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        fold_results = []
        
        print(f"\n{'='*60}")
        print(f"Starting {self.n_folds}-Fold Cross-Validation")
        print(f"{'='*60}\n")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n{'='*60}")
            print(f"Fold {fold}/{self.n_folds}")
            print(f"{'='*60}")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            print(f"Train size: {len(X_train_fold)}, Validation size: {len(X_val_fold)}")
            
            # Build new model for this fold
            model = model_builder()
            
            # Train model
            history = train_func(model, X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            
            # Evaluate on validation set
            val_metrics = evaluate_func(model, X_val_fold, y_val_fold)
            
            # Store results
            fold_result = {
                'fold': fold,
                'train_size': len(X_train_fold),
                'val_size': len(X_val_fold),
                'metrics': val_metrics,
                'history': history.history if hasattr(history, 'history') else history
            }
            fold_results.append(fold_result)
            
            if verbose > 0:
                print(f"\nFold {fold} Results:")
                for metric_name, metric_value in val_metrics.items():
                    print(f"  {metric_name}: {metric_value:.4f}")
        
        # Aggregate results
        self.cv_results = fold_results
        aggregated_results = self._aggregate_results(fold_results)
        
        print(f"\n{'='*60}")
        print("Cross-Validation Summary")
        print(f"{'='*60}")
        for metric_name, values in aggregated_results.items():
            print(f"{metric_name}:")
            print(f"  Mean: {values['mean']:.4f}")
            print(f"  Std:  {values['std']:.4f}")
            print(f"  Min:  {values['min']:.4f}")
            print(f"  Max:  {values['max']:.4f}")
        
        return {
            'fold_results': fold_results,
            'aggregated_results': aggregated_results
        }
    
    def _aggregate_results(self, fold_results: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate metrics across folds.
        
        Args:
            fold_results: List of results from each fold
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Collect all metrics
        metrics_by_name = {}
        for result in fold_results:
            for metric_name, metric_value in result['metrics'].items():
                if metric_name not in metrics_by_name:
                    metrics_by_name[metric_name] = []
                metrics_by_name[metric_name].append(metric_value)
        
        # Calculate statistics
        aggregated = {}
        for metric_name, values in metrics_by_name.items():
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        return aggregated
    
    def save_results(self, filepath: str):
        """
        Save cross-validation results to JSON file.
        
        Args:
            filepath: Path to save results
        """
        # Create parent directory if it doesn't exist
        parent_dir = Path(filepath).parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        results_to_save = []
        for result in self.cv_results:
            serializable_result = {
                'fold': result['fold'],
                'train_size': int(result['train_size']),
                'val_size': int(result['val_size']),
                'metrics': {k: float(v) for k, v in result['metrics'].items()}
            }
            results_to_save.append(serializable_result)
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nCross-validation results saved to {filepath}")
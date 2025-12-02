"""
Data preprocessing utilities for the anti-vax prediction project.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
from .sampling import get_sampler, BaseSampler
from .feature_engineering import FeatureEngineer


class Preprocessor:
    """
    Preprocess data for neural network training.
    """
    
    def __init__(
        self, 
        scale_features: bool = True,
        remove_constant_features: bool = True,
        remove_redundant_features: bool = True,
        sampler: Optional[BaseSampler] = None,
        feature_engineer: Optional[FeatureEngineer] = None
    ):
        """
        Initialize Preprocessor.
        
        Args:
            scale_features: Whether to standardize features
            remove_constant_features: Whether to remove zero-variance features
            remove_redundant_features: Whether to remove redundant predictors
            sampler: Sampling method for handling class imbalance (optional)
            feature_engineer: Feature engineering transformer (optional)
        """
        self.scale_features = scale_features
        self.remove_constant_features = remove_constant_features
        self.remove_redundant_features = remove_redundant_features
        self.sampler = sampler
        self.feature_engineer = feature_engineer
        self.scaler = StandardScaler() if scale_features else None
        self.is_fitted = False
        self.features_to_remove = []
        self.constant_features = []
        
        # Pre-identified constant features from data analysis
        self.known_constant_features = [
            'Q312_1', 'Q312_2', 'Q54_3', 'Q64_1', 'Q64_2', 'Q64_3',
            'Q65_1', 'Q66_3', 'Q67_2', 'Q67_3', 'Q69_1', 'Q69_2', 'Q69_3'
        ]
        self.features_to_remove = []
        self.constant_features = []
        
        # Pre-identified constant features from data analysis
        self.known_constant_features = [
            'Q312_1', 'Q312_2', 'Q54_3', 'Q64_1', 'Q64_2', 'Q64_3',
            'Q65_1', 'Q66_3', 'Q67_2', 'Q67_3', 'Q69_1', 'Q69_2', 'Q69_3'
        ]
    
    def _identify_redundant_features(self, X: pd.DataFrame) -> List[str]:
        """
        Identify redundant features to remove.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            List of feature names to remove
        """
        features_to_remove = []
        
        # Remove known constant features
        if self.remove_constant_features:
            for col in self.known_constant_features:
                if col in X.columns:
                    features_to_remove.append(col)
            
            # Also check for any additional constant features
            for col in X.columns:
                if col not in features_to_remove:
                    if X[col].nunique() == 1:
                        features_to_remove.append(col)
                        self.constant_features.append(col)
        
        # Remove redundant missing indicators (all zeros)
        if self.remove_redundant_features:
            missing_indicators = [col for col in X.columns if col.endswith('_missing')]
            for col in missing_indicators:
                if col not in features_to_remove:
                    if X[col].sum() == 0:  # All zeros - no missing values
                        features_to_remove.append(col)
        
        return features_to_remove
    
    def _remove_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove redundant features from DataFrame.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with redundant features removed
        """
        if not self.features_to_remove:
            return X
        
        features_to_drop = [f for f in self.features_to_remove if f in X.columns]
        if features_to_drop:
            print(f"\nRemoving {len(features_to_drop)} redundant features:")
            for feat in features_to_drop:
                reason = "constant" if feat in self.constant_features or feat in self.known_constant_features else "redundant"
                print(f"  - {feat} ({reason})")
            X = X.drop(columns=features_to_drop)
        
        return X
        
    def fit(self, X: pd.DataFrame) -> 'Preprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Self for chaining
        """
        # Apply feature engineering first (before removing redundant features)
        if self.feature_engineer is not None:
            X = self.feature_engineer.fit_transform(X)
        
        # Identify redundant features
        self.features_to_remove = self._identify_redundant_features(X)
        
        # Remove redundant features
        X_cleaned = self._remove_features(X)
        
        if self.scale_features:
            self.scaler.fit(X_cleaned)
            self.is_fitted = True
            print(f"\nScaler fitted on {X_cleaned.shape[1]} features (after removing redundant)")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed feature array
        """
        # Apply feature engineering first
        if self.feature_engineer is not None:
            X = self.feature_engineer.transform(X)
        
        # Remove redundant features
        X_cleaned = self._remove_features(X)
        
        if self.scale_features:
            if not self.is_fitted:
                raise RuntimeError("Preprocessor must be fitted before transform")
            return self.scaler.transform(X_cleaned)
        return X_cleaned.values
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed feature array
        """
        # Apply feature engineering first
        if self.feature_engineer is not None:
            X = self.feature_engineer.fit_transform(X)
        
        # Identify redundant features
        self.features_to_remove = self._identify_redundant_features(X)
        
        # Remove redundant features
        X_cleaned = self._remove_features(X)
        
        if self.scale_features:
            transformed = self.scaler.fit_transform(X_cleaned)
            self.is_fitted = True
            print(f"\nScaler fitted and data transformed")
            print(f"Final feature count: {X_cleaned.shape[1]} (removed {len(self.features_to_remove)})")
            return transformed
        return X_cleaned.values
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        apply_sampling: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by splitting, sampling, and scaling.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            apply_sampling: Whether to apply sampling to training data
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Split data first (before sampling to avoid data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nOriginal split:")
        print(f"  Training set size: {len(X_train)}")
        print(f"  Test set size: {len(X_test)}")
        
        # Fit and transform training data (identify redundant features and scale)
        X_train_scaled = self.fit_transform(X_train)
        X_test_scaled = self.transform(X_test)
        y_train_arr = y_train.values
        y_test_arr = y_test.values
        
        # Apply sampling to training data only (after scaling)
        if apply_sampling and self.sampler is not None:
            print(f"\nApplying {self.sampler.__class__.__name__}...")
            X_train_scaled, y_train_arr = self.sampler.fit_resample(
                X_train_scaled, y_train_arr
            )
            print(f"  Training set size after sampling: {len(X_train_scaled)}")
        
        return X_train_scaled, X_test_scaled, y_train_arr, y_test_arr
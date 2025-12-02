"""
Advanced Feature Engineering for Anti-Vax Prediction

This module creates new features from existing ones to improve model performance:
- Domain-specific features (survey question aggregations)
- Statistical features (mean, std, min, max across feature groups)
- Interaction features (ratios, products)
- Count-based features
"""
import numpy as np
import pandas as pd
from typing import Optional, List


class FeatureEngineer:
    """
    Advanced feature engineering for survey data.
    
    Creates domain-specific and statistical features to enhance
    the predictive power of the model.
    """
    
    def __init__(
        self,
        enable_domain_features: bool = True,
        enable_statistical_features: bool = True,
        enable_interaction_features: bool = True,
        enable_count_features: bool = True,
        selected_features: Optional[List[str]] = None
    ):
        """
        Initialize FeatureEngineer.
        
        Args:
            enable_domain_features: Create survey question group aggregations
            enable_statistical_features: Create statistical summaries
            enable_interaction_features: Create feature interactions
            enable_count_features: Create count-based features
            selected_features: List of features to use (None = use all)
        """
        self.enable_domain_features = enable_domain_features
        self.enable_statistical_features = enable_statistical_features
        self.enable_interaction_features = enable_interaction_features
        self.enable_count_features = enable_count_features
        self.selected_features = selected_features
        self.feature_names_ = None
        
    def create_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features based on survey question groups.
        
        Survey questions are grouped (Q41_*, Q42_*, etc.) and we create
        aggregations within each group.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with domain features added
        """
        X_new = X.copy()
        
        # Q41_* questions (8 features) - Vaccine-related attitudes
        q41_cols = [c for c in X.columns if c.startswith('Q41_')]
        if q41_cols and len(q41_cols) > 1:
            X_new['Q41_mean'] = X[q41_cols].mean(axis=1)
            X_new['Q41_std'] = X[q41_cols].std(axis=1)
            X_new['Q41_max'] = X[q41_cols].max(axis=1)
            X_new['Q41_min'] = X[q41_cols].min(axis=1)
            X_new['Q41_range'] = X_new['Q41_max'] - X_new['Q41_min']
            X_new['Q41_median'] = X[q41_cols].median(axis=1)
        
        # Q42_* questions (7 features) - Health beliefs
        q42_cols = [c for c in X.columns if c.startswith('Q42_')]
        if q42_cols and len(q42_cols) > 1:
            X_new['Q42_mean'] = X[q42_cols].mean(axis=1)
            X_new['Q42_std'] = X[q42_cols].std(axis=1)
            X_new['Q42_max'] = X[q42_cols].max(axis=1)
            X_new['Q42_min'] = X[q42_cols].min(axis=1)
            X_new['Q42_range'] = X_new['Q42_max'] - X_new['Q42_min']
        
        # Q43_* questions (15 features) - Vaccine knowledge
        q43_cols = [c for c in X.columns if c.startswith('Q43_')]
        if q43_cols and len(q43_cols) > 1:
            X_new['Q43_mean'] = X[q43_cols].mean(axis=1)
            X_new['Q43_std'] = X[q43_cols].std(axis=1)
            X_new['Q43_sum'] = X[q43_cols].sum(axis=1)
            X_new['Q43_max'] = X[q43_cols].max(axis=1)
            X_new['Q43_min'] = X[q43_cols].min(axis=1)
            X_new['Q43_median'] = X[q43_cols].median(axis=1)
        
        # Q310_* questions (attitude items)
        q310_cols = [c for c in X.columns if c.startswith('Q310_')]
        if q310_cols and len(q310_cols) > 1:
            X_new['Q310_mean'] = X[q310_cols].mean(axis=1)
            X_new['Q310_std'] = X[q310_cols].std(axis=1)
            X_new['Q310_max'] = X[q310_cols].max(axis=1)
        
        # Q44_* questions
        q44_cols = [c for c in X.columns if c.startswith('Q44_')]
        if q44_cols and len(q44_cols) > 1:
            X_new['Q44_mean'] = X[q44_cols].mean(axis=1)
            X_new['Q44_std'] = X[q44_cols].std(axis=1)
        
        # Q45_* questions
        q45_cols = [c for c in X.columns if c.startswith('Q45_')]
        if q45_cols and len(q45_cols) > 1:
            X_new['Q45_mean'] = X[q45_cols].mean(axis=1)
            X_new['Q45_std'] = X[q45_cols].std(axis=1)
        
        # Q5_* questions (demographic)
        q5_cols = [c for c in X.columns if c.startswith('Q5_')]
        if q5_cols and len(q5_cols) > 1:
            X_new['Q5_sum'] = X[q5_cols].sum(axis=1)
        
        # Q6_* questions
        q6_cols = [c for c in X.columns if c.startswith('Q6_')]
        if q6_cols and len(q6_cols) > 1:
            X_new['Q6_mean'] = X[q6_cols].mean(axis=1)
        
        return X_new
    
    def create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical aggregation features across all columns.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with statistical features added
        """
        X_new = X.copy()
        
        # Overall statistics across all features
        X_new['overall_mean'] = X.mean(axis=1)
        X_new['overall_std'] = X.std(axis=1)
        X_new['overall_max'] = X.max(axis=1)
        X_new['overall_min'] = X.min(axis=1)
        X_new['overall_median'] = X.median(axis=1)
        X_new['overall_range'] = X_new['overall_max'] - X_new['overall_min']
        
        # Quartiles
        X_new['overall_q25'] = X.quantile(0.25, axis=1)
        X_new['overall_q75'] = X.quantile(0.75, axis=1)
        X_new['overall_iqr'] = X_new['overall_q75'] - X_new['overall_q25']
        
        # Coefficient of variation (std/mean)
        X_new['overall_cv'] = X_new['overall_std'] / (X_new['overall_mean'] + 1e-6)
        
        # Skewness proxy (mean - median)
        X_new['overall_skew_proxy'] = X_new['overall_mean'] - X_new['overall_median']
        
        return X_new
    
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features (ratios, products).
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with interaction features added
        """
        X_new = X.copy()
        
        # Key single features for interactions
        key_features = []
        
        # Identify key binary/categorical features
        for col in X.columns:
            if col.startswith('Q21') or col.startswith('Q29') or col.startswith('Q210'):
                key_features.append(col)
        
        # Ratios between aggregated features (if they exist)
        if 'Q41_mean' in X_new.columns and 'Q42_mean' in X_new.columns:
            X_new['Q41_Q42_ratio'] = X_new['Q41_mean'] / (X_new['Q42_mean'] + 1e-6)
        
        if 'Q41_mean' in X_new.columns and 'Q43_mean' in X_new.columns:
            X_new['Q41_Q43_ratio'] = X_new['Q41_mean'] / (X_new['Q43_mean'] + 1e-6)
        
        if 'Q42_mean' in X_new.columns and 'Q43_mean' in X_new.columns:
            X_new['Q42_Q43_ratio'] = X_new['Q42_mean'] / (X_new['Q43_mean'] + 1e-6)
        
        # Products of key features
        if 'Q41_std' in X_new.columns and 'Q42_std' in X_new.columns:
            X_new['Q41_Q42_std_product'] = X_new['Q41_std'] * X_new['Q42_std']
        
        # Interaction with overall statistics
        if 'Q41_mean' in X_new.columns:
            X_new['Q41_vs_overall'] = X_new['Q41_mean'] - X_new['overall_mean']
        
        if 'Q42_mean' in X_new.columns:
            X_new['Q42_vs_overall'] = X_new['Q42_mean'] - X_new['overall_mean']
        
        return X_new
    
    def create_count_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create count-based features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with count features added
        """
        X_new = X.copy()
        
        # Count of specific values (common in Likert scales)
        X_new['count_1s'] = (X == 1).sum(axis=1)
        X_new['count_2s'] = (X == 2).sum(axis=1)
        X_new['count_3s'] = (X == 3).sum(axis=1)
        X_new['count_4s'] = (X == 4).sum(axis=1)
        X_new['count_5s'] = (X == 5).sum(axis=1)
        
        # Count zeros (if applicable)
        X_new['count_zeros'] = (X == 0).sum(axis=1)
        
        # Count of max values
        X_new['count_max_values'] = (X == X.max(axis=1).values.reshape(-1, 1)).sum(axis=1)
        
        # Count of min values
        X_new['count_min_values'] = (X == X.min(axis=1).values.reshape(-1, 1)).sum(axis=1)
        
        # Proportion of values above/below median
        median_vals = X.median(axis=1).values.reshape(-1, 1)
        X_new['prop_above_median'] = (X > median_vals).sum(axis=1) / X.shape[1]
        X_new['prop_below_median'] = (X < median_vals).sum(axis=1) / X.shape[1]
        
        return X_new
    
    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureEngineer':
        """
        Fit the feature engineer (learn feature names).
        
        Args:
            X: Feature DataFrame
            y: Target (not used, for sklearn compatibility)
            
        Returns:
            Self for chaining
        """
        # Apply feature selection if specified
        if self.selected_features is not None:
            X_selected = X[self.selected_features]
        else:
            X_selected = X
        
        # Create features to learn the final feature names
        X_transformed = X_selected.copy()
        
        if self.enable_domain_features:
            X_transformed = self.create_domain_features(X_transformed)
        
        if self.enable_statistical_features:
            X_transformed = self.create_statistical_features(X_transformed)
        
        if self.enable_interaction_features:
            X_transformed = self.create_interaction_features(X_transformed)
        
        if self.enable_count_features:
            X_transformed = self.create_count_features(X_transformed)
        
        self.feature_names_ = X_transformed.columns.tolist()
        
        print(f"\nFeature Engineering Summary:")
        print(f"  Original features: {X.shape[1]}")
        print(f"  Engineered features: {len(self.feature_names_)}")
        print(f"  New features created: {len(self.feature_names_) - X.shape[1]}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features by creating engineered features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed DataFrame with engineered features
        """
        # Apply feature selection if specified
        if self.selected_features is not None:
            X_selected = X[self.selected_features]
        else:
            X_selected = X
        
        X_transformed = X_selected.copy()
        
        if self.enable_domain_features:
            X_transformed = self.create_domain_features(X_transformed)
        
        if self.enable_statistical_features:
            X_transformed = self.create_statistical_features(X_transformed)
        
        if self.enable_interaction_features:
            X_transformed = self.create_interaction_features(X_transformed)
        
        if self.enable_count_features:
            X_transformed = self.create_count_features(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature DataFrame
            y: Target (not used, for sklearn compatibility)
            
        Returns:
            Transformed DataFrame with engineered features
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names.
        
        Returns:
            List of feature names after transformation
        """
        if self.feature_names_ is None:
            raise RuntimeError("FeatureEngineer must be fitted first")
        return self.feature_names_

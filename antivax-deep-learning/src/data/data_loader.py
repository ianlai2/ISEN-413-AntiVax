"""
Data loading utilities for the anti-vax prediction project.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """
    Load and prepare data for model training.
    """
    
    def __init__(self, data_path: str, target_column: str = "Y"):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the training data CSV file
            target_column: Name of the target column
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.data = None
        self.features = None
        self.target = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            Loaded DataFrame
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded data with shape: {self.data.shape}")
        return self.data
    
    def split_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target.
        
        Returns:
            Tuple of (features, target)
        """
        if self.data is None:
            self.load_data()
        
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        self.features = self.data.drop(columns=[self.target_column])
        self.target = self.data[self.target_column]
        
        print(f"Features shape: {self.features.shape}")
        print(f"Target shape: {self.target.shape}")
        print(f"Target distribution:\n{self.target.value_counts(normalize=True)}")
        
        return self.features, self.target
    
    def get_feature_names(self) -> list:
        """
        Get list of feature names.
        
        Returns:
            List of feature column names
        """
        if self.features is None:
            self.split_features_target()
        return self.features.columns.tolist()
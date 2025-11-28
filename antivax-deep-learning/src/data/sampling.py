"""
Sampling methods for handling class imbalance.
Includes SMOTE, random oversampling, random undersampling, and combined methods.
"""
import numpy as np
from typing import Tuple, Optional, Literal
from collections import Counter


class BaseSampler:
    """Base class for sampling methods."""
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize sampler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample the dataset.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        raise NotImplementedError("Subclasses must implement fit_resample")


class RandomOverSampler(BaseSampler):
    """
    Random oversampling of minority class.
    Randomly duplicates minority class samples to balance classes.
    """
    
    def __init__(
        self,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None
    ):
        """
        Initialize RandomOverSampler.
        
        Args:
            sampling_strategy: 'auto' balances to majority, 'minority' balances to mean
            random_state: Random seed
        """
        super().__init__(random_state)
        self.sampling_strategy = sampling_strategy
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample by randomly oversampling minority class.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        if len(unique_classes) != 2:
            raise ValueError("Currently only supports binary classification")
        
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        
        majority_count = np.max(class_counts)
        minority_count = np.min(class_counts)
        
        # Determine target count for minority class
        if self.sampling_strategy == 'auto':
            target_count = majority_count
        else:
            target_count = int((majority_count + minority_count) / 2)
        
        # Get indices for each class
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Calculate how many samples to add
        n_samples_to_add = target_count - minority_count
        
        print(f"\nRandom Oversampling:")
        print(f"  Original class distribution: {dict(zip(unique_classes, class_counts))}")
        print(f"  Minority class ({minority_class}): {minority_count} → {target_count}")
        print(f"  Adding {n_samples_to_add} samples")
        
        # Randomly sample from minority class with replacement
        oversampled_indices = np.random.choice(
            minority_indices,
            size=n_samples_to_add,
            replace=True
        )
        
        # Combine original majority + minority + oversampled minority
        all_indices = np.concatenate([majority_indices, minority_indices, oversampled_indices])
        
        # Shuffle
        np.random.shuffle(all_indices)
        
        X_resampled = X[all_indices]
        y_resampled = y[all_indices]
        
        new_counts = dict(zip(*np.unique(y_resampled, return_counts=True)))
        print(f"  New class distribution: {new_counts}")
        
        return X_resampled, y_resampled


class RandomUnderSampler(BaseSampler):
    """
    Random undersampling of majority class.
    Randomly removes majority class samples to balance classes.
    """
    
    def __init__(
        self,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None
    ):
        """
        Initialize RandomUnderSampler.
        
        Args:
            sampling_strategy: 'auto' balances to minority, 'majority' balances to mean
            random_state: Random seed
        """
        super().__init__(random_state)
        self.sampling_strategy = sampling_strategy
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample by randomly undersampling majority class.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        if len(unique_classes) != 2:
            raise ValueError("Currently only supports binary classification")
        
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        
        majority_count = np.max(class_counts)
        minority_count = np.min(class_counts)
        
        # Determine target count for majority class
        if self.sampling_strategy == 'auto':
            target_count = minority_count
        else:
            target_count = int((majority_count + minority_count) / 2)
        
        # Get indices for each class
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        print(f"\nRandom Undersampling:")
        print(f"  Original class distribution: {dict(zip(unique_classes, class_counts))}")
        print(f"  Majority class ({majority_class}): {majority_count} → {target_count}")
        print(f"  Removing {majority_count - target_count} samples")
        
        # Randomly sample from majority class without replacement
        undersampled_indices = np.random.choice(
            majority_indices,
            size=target_count,
            replace=False
        )
        
        # Combine minority + undersampled majority
        all_indices = np.concatenate([minority_indices, undersampled_indices])
        
        # Shuffle
        np.random.shuffle(all_indices)
        
        X_resampled = X[all_indices]
        y_resampled = y[all_indices]
        
        new_counts = dict(zip(*np.unique(y_resampled, return_counts=True)))
        print(f"  New class distribution: {new_counts}")
        
        return X_resampled, y_resampled


class SMOTE(BaseSampler):
    """
    Synthetic Minority Over-sampling Technique.
    Generates synthetic samples by interpolating between minority class samples.
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None
    ):
        """
        Initialize SMOTE.
        
        Args:
            k_neighbors: Number of nearest neighbors to use
            sampling_strategy: 'auto' balances to majority, 'minority' balances to mean
            random_state: Random seed
        """
        super().__init__(random_state)
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
    
    def _find_k_nearest_neighbors(
        self,
        X_minority: np.ndarray,
        sample_idx: int,
        k: int
    ) -> np.ndarray:
        """
        Find k nearest neighbors for a sample.
        
        Args:
            X_minority: Minority class samples
            sample_idx: Index of the sample
            k: Number of neighbors
            
        Returns:
            Indices of k nearest neighbors
        """
        # Calculate Euclidean distances
        sample = X_minority[sample_idx]
        distances = np.sqrt(np.sum((X_minority - sample) ** 2, axis=1))
        
        # Exclude the sample itself
        distances[sample_idx] = np.inf
        
        # Get k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        
        return nearest_indices
    
    def _generate_synthetic_sample(
        self,
        sample: np.ndarray,
        neighbor: np.ndarray
    ) -> np.ndarray:
        """
        Generate a synthetic sample between two samples.
        
        Args:
            sample: Original sample
            neighbor: Neighbor sample
            
        Returns:
            Synthetic sample
        """
        # Random interpolation factor between 0 and 1
        alpha = np.random.random()
        
        # Linear interpolation
        synthetic = sample + alpha * (neighbor - sample)
        
        return synthetic
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample using SMOTE algorithm.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        if len(unique_classes) != 2:
            raise ValueError("Currently only supports binary classification")
        
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        
        majority_count = np.max(class_counts)
        minority_count = np.min(class_counts)
        
        # Determine target count for minority class
        if self.sampling_strategy == 'auto':
            target_count = majority_count
        else:
            target_count = int((majority_count + minority_count) / 2)
        
        # Calculate number of synthetic samples needed
        n_synthetic = target_count - minority_count
        
        print(f"\nSMOTE Oversampling:")
        print(f"  Original class distribution: {dict(zip(unique_classes, class_counts))}")
        print(f"  Minority class ({minority_class}): {minority_count} → {target_count}")
        print(f"  Generating {n_synthetic} synthetic samples")
        print(f"  Using k={self.k_neighbors} nearest neighbors")
        
        # Get minority class samples
        minority_mask = y == minority_class
        X_minority = X[minority_mask]
        X_majority = X[~minority_mask]
        
        # Check if we have enough neighbors
        if len(X_minority) < self.k_neighbors + 1:
            print(f"  Warning: Only {len(X_minority)} minority samples, reducing k to {len(X_minority) - 1}")
            k = len(X_minority) - 1
        else:
            k = self.k_neighbors
        
        # Generate synthetic samples
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            # Randomly select a minority sample
            sample_idx = np.random.randint(0, len(X_minority))
            sample = X_minority[sample_idx]
            
            # Find k nearest neighbors
            neighbor_indices = self._find_k_nearest_neighbors(X_minority, sample_idx, k)
            
            # Randomly select one neighbor
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = X_minority[neighbor_idx]
            
            # Generate synthetic sample
            synthetic = self._generate_synthetic_sample(sample, neighbor)
            synthetic_samples.append(synthetic)
        
        synthetic_samples = np.array(synthetic_samples)
        
        # Combine original data with synthetic samples
        X_resampled = np.vstack([X, synthetic_samples])
        y_minority_synthetic = np.full(n_synthetic, minority_class)
        y_resampled = np.concatenate([y, y_minority_synthetic])
        
        # Shuffle
        shuffle_indices = np.random.permutation(len(X_resampled))
        X_resampled = X_resampled[shuffle_indices]
        y_resampled = y_resampled[shuffle_indices]
        
        new_counts = dict(zip(*np.unique(y_resampled, return_counts=True)))
        print(f"  New class distribution: {new_counts}")
        
        return X_resampled, y_resampled


class SMOTETomek(BaseSampler):
    """
    SMOTE + Tomek Links.
    Combines SMOTE oversampling with Tomek links undersampling.
    Removes Tomek links (borderline samples) after SMOTE.
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None
    ):
        """
        Initialize SMOTETomek.
        
        Args:
            k_neighbors: Number of neighbors for SMOTE
            sampling_strategy: Sampling strategy
            random_state: Random seed
        """
        super().__init__(random_state)
        self.smote = SMOTE(k_neighbors, sampling_strategy, random_state)
    
    def _find_tomek_links(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find Tomek links in the dataset.
        
        A Tomek link is a pair of samples from different classes
        that are each other's nearest neighbors.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Boolean mask of samples to keep (True = keep, False = remove)
        """
        n_samples = len(X)
        keep_mask = np.ones(n_samples, dtype=bool)
        
        # Find nearest neighbor for each sample
        for i in range(n_samples):
            distances = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
            distances[i] = np.inf  # Exclude self
            nearest_idx = np.argmin(distances)
            
            # Check if they form a Tomek link (different classes and mutual nearest neighbors)
            if y[i] != y[nearest_idx]:
                # Check if i is nearest neighbor of nearest_idx
                distances_reverse = np.sqrt(np.sum((X - X[nearest_idx]) ** 2, axis=1))
                distances_reverse[nearest_idx] = np.inf
                reverse_nearest = np.argmin(distances_reverse)
                
                if reverse_nearest == i:
                    # Tomek link found - mark majority class sample for removal
                    unique_classes, class_counts = np.unique(y, return_counts=True)
                    majority_class = unique_classes[np.argmax(class_counts)]
                    
                    if y[i] == majority_class:
                        keep_mask[i] = False
                    elif y[nearest_idx] == majority_class:
                        keep_mask[nearest_idx] = False
        
        return keep_mask
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample using SMOTE + Tomek links.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        print("\nSMOTE + Tomek Links:")
        
        # First apply SMOTE
        X_smote, y_smote = self.smote.fit_resample(X, y)
        
        # Then remove Tomek links
        print("\n  Removing Tomek links...")
        keep_mask = self._find_tomek_links(X_smote, y_smote)
        n_removed = np.sum(~keep_mask)
        
        X_resampled = X_smote[keep_mask]
        y_resampled = y_smote[keep_mask]
        
        print(f"  Removed {n_removed} Tomek link samples")
        new_counts = dict(zip(*np.unique(y_resampled, return_counts=True)))
        print(f"  Final class distribution: {new_counts}")
        
        return X_resampled, y_resampled


class SMOTEENN(BaseSampler):
    """
    SMOTE + Edited Nearest Neighbors.
    Combines SMOTE oversampling with ENN cleaning.
    Removes samples whose class differs from majority of k neighbors.
    """
    
    def __init__(
        self,
        k_neighbors_smote: int = 5,
        k_neighbors_enn: int = 3,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None
    ):
        """
        Initialize SMOTEENN.
        
        Args:
            k_neighbors_smote: Number of neighbors for SMOTE
            k_neighbors_enn: Number of neighbors for ENN
            sampling_strategy: Sampling strategy
            random_state: Random seed
        """
        super().__init__(random_state)
        self.smote = SMOTE(k_neighbors_smote, sampling_strategy, random_state)
        self.k_neighbors_enn = k_neighbors_enn
    
    def _edited_nearest_neighbors(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply Edited Nearest Neighbors cleaning.
        
        Remove samples whose class differs from the majority
        of their k nearest neighbors.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Boolean mask of samples to keep
        """
        n_samples = len(X)
        keep_mask = np.ones(n_samples, dtype=bool)
        
        for i in range(n_samples):
            # Find k nearest neighbors
            distances = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
            distances[i] = np.inf
            nearest_indices = np.argsort(distances)[:self.k_neighbors_enn]
            
            # Get majority class of neighbors
            neighbor_classes = y[nearest_indices]
            majority_class = np.bincount(neighbor_classes.astype(int)).argmax()
            
            # Remove if sample class != majority class of neighbors
            if y[i] != majority_class:
                keep_mask[i] = False
        
        return keep_mask
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample using SMOTE + ENN.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        print("\nSMOTE + ENN:")
        
        # First apply SMOTE
        X_smote, y_smote = self.smote.fit_resample(X, y)
        
        # Then apply ENN cleaning
        print(f"\n  Applying ENN cleaning (k={self.k_neighbors_enn})...")
        keep_mask = self._edited_nearest_neighbors(X_smote, y_smote)
        n_removed = np.sum(~keep_mask)
        
        X_resampled = X_smote[keep_mask]
        y_resampled = y_smote[keep_mask]
        
        print(f"  Removed {n_removed} noisy samples")
        new_counts = dict(zip(*np.unique(y_resampled, return_counts=True)))
        print(f"  Final class distribution: {new_counts}")
        
        return X_resampled, y_resampled


def get_sampler(
    method: Literal['none', 'oversample', 'undersample', 'smote', 'smote_tomek', 'smote_enn'] = 'none',
    **kwargs
) -> Optional[BaseSampler]:
    """
    Factory function to get a sampler instance.
    
    Args:
        method: Sampling method to use
        **kwargs: Additional arguments for the sampler
        
    Returns:
        Sampler instance or None if method='none'
    """
    if method == 'none':
        return None
    elif method == 'oversample':
        return RandomOverSampler(**kwargs)
    elif method == 'undersample':
        return RandomUnderSampler(**kwargs)
    elif method == 'smote':
        return SMOTE(**kwargs)
    elif method == 'smote_tomek':
        return SMOTETomek(**kwargs)
    elif method == 'smote_enn':
        return SMOTEENN(**kwargs)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

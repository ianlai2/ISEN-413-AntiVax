"""
Model training utilities.
"""
import numpy as np
from tensorflow import keras
from pathlib import Path
from typing import Optional, Dict, Any


class Trainer:
    """
    Handle model training with callbacks and monitoring.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        early_stopping_patience: int = 15,
        verbose: int = 1
    ):
        """
        Initialize Trainer.
        
        Args:
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            validation_split: Fraction of training data to use for validation
            early_stopping_patience: Number of epochs with no improvement to wait
            verbose: Verbosity level
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        
    def get_callbacks(
        self,
        model_checkpoint_path: Optional[str] = None
    ) -> list:
        """
        Create training callbacks.
        
        Args:
            model_checkpoint_path: Path to save best model
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        if model_checkpoint_path:
            Path(model_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            checkpoint = keras.callbacks.ModelCheckpoint(
                model_checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def train(
        self,
        model: keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_checkpoint_path: Optional[str] = None
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            model: Keras model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            model_checkpoint_path: Path to save best model
            
        Returns:
            Training history
        """
        callbacks = self.get_callbacks(model_checkpoint_path)
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0.0
        else:
            validation_data = None
            validation_split = self.validation_split
        
        print("\nStarting training...")
        history = model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        print(f"\nTraining completed. Total epochs: {len(history.history['loss'])}")
        
        return history


def train_model_wrapper(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any]
) -> keras.callbacks.History:
    """
    Wrapper function for training models in cross-validation.
    
    Args:
        model: Keras model to train
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Configuration dictionary
        
    Returns:
        Training history
    """
    training_config = config.get('training', {})
    
    trainer = Trainer(
        batch_size=training_config.get('batch_size', 32),
        epochs=training_config.get('epochs', 100),
        validation_split=0.0,  # Using explicit validation data
        early_stopping_patience=training_config.get('early_stopping_patience', 15),
        verbose=0  # Less verbose for cross-validation
    )
    
    return trainer.train(model, X_train, y_train, X_val, y_val)
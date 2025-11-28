"""
Model building utilities for creating neural networks.
"""
from typing import Dict, Any
from src.models.neural_network import AntiVaxNN


def build_model_from_config(config: Dict[str, Any], input_dim: int) -> AntiVaxNN:
    """
    Build a neural network model from configuration.
    
    Args:
        config: Configuration dictionary
        input_dim: Number of input features
        
    Returns:
        AntiVaxNN instance with built model
    """
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    nn = AntiVaxNN(
        input_dim=input_dim,
        hidden_layers=model_config.get('hidden_layers', [128, 64, 32]),
        dropout_rate=model_config.get('dropout_rate', 0.3),
        activation=model_config.get('activation', 'relu'),
        output_activation=model_config.get('output_activation', 'sigmoid'),
        learning_rate=training_config.get('learning_rate', 0.001)
    )
    
    nn.build()
    return nn


def save_model(model, filepath: str):
    """
    Save trained model to disk.
    
    Args:
        model: Keras model to save
        filepath: Path where model should be saved
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded Keras model
    """
    from tensorflow import keras
    model = keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model
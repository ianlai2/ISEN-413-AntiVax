"""
Neural network architecture for anti-vax prediction.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import List, Optional


class AntiVaxNN:
    """
    Deep neural network for predicting anti-vax sentiment.
    Includes dropout layers for overfitting prevention.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        output_activation: str = 'sigmoid',
        learning_rate: float = 0.001
    ):
        """
        Initialize neural network.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of neurons in each hidden layer
            dropout_rate: Dropout rate for regularization
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.model = None
        
    def build(self) -> keras.Model:
        """
        Build the neural network architecture.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential(name='AntiVaxPredictor')
        
        # Input layer
        model.add(layers.Input(shape=(self.input_dim,)))
        
        # Hidden layers with dropout
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                units,
                activation=self.activation,
                kernel_initializer='he_normal',
                name=f'hidden_{i+1}'
            ))
            # Add dropout after each hidden layer for regularization
            model.add(layers.Dropout(
                self.dropout_rate,
                name=f'dropout_{i+1}'
            ))
        
        # Output layer (binary classification)
        model.add(layers.Dense(
            1,
            activation=self.output_activation,
            name='output'
        ))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def get_model(self) -> keras.Model:
        """
        Get the built model.
        
        Returns:
            Keras model
        """
        if self.model is None:
            self.build()
        return self.model
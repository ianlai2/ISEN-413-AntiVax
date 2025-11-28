"""
Seed management for reproducible results.
"""
import random
import numpy as np
import tensorflow as tf
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Configure TensorFlow for deterministic operations
    tf.config.experimental.enable_op_determinism()
    
    print(f"Random seed set to {seed} for reproducibility")
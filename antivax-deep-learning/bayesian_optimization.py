"""
Bayesian Hyperparameter Optimization for Neural Network

This module uses Bayesian optimization to efficiently search for optimal
hyperparameters. More efficient than grid search, it uses a probabilistic
model to guide the search toward promising regions of the parameter space.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scikeras.wrappers import KerasClassifier
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.sampling import get_sampler


def create_model_for_bayes(hidden_layer_1=128, hidden_layer_2=64, hidden_layer_3=32,
                           hidden_layer_4=0, hidden_layer_5=0,
                           dropout_rate=0.3, learning_rate=0.001, optimizer='adam',
                           meta=None):
    """
    Model builder for Bayesian optimization.
    
    Args:
        hidden_layer_1: Neurons in first hidden layer
        hidden_layer_2: Neurons in second hidden layer
        hidden_layer_3: Neurons in third hidden layer
        hidden_layer_4: Neurons in fourth hidden layer (0 to disable)
        hidden_layer_5: Neurons in fifth hidden layer (0 to disable)
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type ('adam', 'adamw', 'rmsprop')
        meta: Metadata dict containing input_dim
        
    Returns:
        Compiled Keras model
    """
    if meta is None:
        raise ValueError("meta dict with 'n_features_in_' is required")
    
    input_dim = meta['n_features_in_']
    
    # Build layers dynamically based on which layers are enabled (non-zero)
    layers = [
        keras.layers.Dense(hidden_layer_1, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(hidden_layer_2, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(hidden_layer_3, activation='relu'),
        keras.layers.Dropout(dropout_rate),
    ]
    
    # Add optional fourth layer if specified
    if hidden_layer_4 > 0:
        layers.extend([
            keras.layers.Dense(hidden_layer_4, activation='relu'),
            keras.layers.Dropout(dropout_rate)
        ])
    
    # Add optional fifth layer if specified
    if hidden_layer_5 > 0:
        layers.extend([
            keras.layers.Dense(hidden_layer_5, activation='relu'),
            keras.layers.Dropout(dropout_rate)
        ])
    
    # Output layer
    layers.append(keras.layers.Dense(1, activation='sigmoid'))
    
    model = keras.Sequential(layers)
    
    # Select optimizer
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'adamw':
        opt = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    return model


def plot_optimization_results(search_result, save_path='results/plots/bayesian_optimization.png'):
    """
    Visualize Bayesian optimization results.
    
    Args:
        search_result: BayesSearchCV fitted object
        save_path: Path to save plot
    """
    # Extract results
    results = pd.DataFrame(search_result.cv_results_)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Convergence plot
    ax = axes[0, 0]
    best_scores = []
    current_best = -np.inf
    for score in results['mean_test_score']:
        if score > current_best:
            current_best = score
        best_scores.append(current_best)
    
    ax.plot(best_scores, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Score (− Misclassification Error)', fontsize=12)
    ax.set_title('Bayesian Optimization Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Parameter importance (based on variance)
    ax = axes[0, 1]
    param_cols = [col for col in results.columns if col.startswith('param_')]
    param_importance = {}
    
    for param in param_cols:
        param_name = param.replace('param_', '')
        # Calculate correlation with score
        if results[param].dtype in [np.float64, np.int64]:
            corr = np.abs(results[param].corr(results['mean_test_score']))
            param_importance[param_name] = corr
    
    if param_importance:
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        params, importances = zip(*sorted_params)
        
        ax.barh(params, importances, color='steelblue')
        ax.set_xlabel('Correlation with Score', fontsize=12)
        ax.set_title('Parameter Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Score distribution
    ax = axes[1, 0]
    ax.hist(results['mean_test_score'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(search_result.best_score_, color='red', linestyle='--', linewidth=2, 
               label=f'Best: {search_result.best_score_:.4f}')
    ax.set_xlabel('Score (− Misclassification Error)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Top 10 configurations
    ax = axes[1, 1]
    top_10 = results.nlargest(10, 'mean_test_score')[['mean_test_score', 'std_test_score']]
    top_10_idx = range(1, len(top_10) + 1)
    
    ax.errorbar(top_10_idx, top_10['mean_test_score'], 
                yerr=top_10['std_test_score'], 
                fmt='o-', linewidth=2, markersize=8, capsize=5, color='steelblue')
    ax.set_xlabel('Configuration Rank', fontsize=12)
    ax.set_ylabel('Score (− Misclassification Error)', fontsize=12)
    ax.set_title('Top 10 Configurations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(top_10_idx)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nOptimization visualization saved to: {save_path}")
    plt.close()


def run_bayesian_optimization(n_iter=30, cv_folds=5, n_jobs=1, random_state=42):
    """
    Run Bayesian hyperparameter optimization.
    
    Args:
        n_iter: Number of optimization iterations
        cv_folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs (-1 for all cores)
        random_state: Random seed for reproducibility
        
    Returns:
        BayesSearchCV fitted object
    """
    print("="*70)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(random_state)
    
    # Load configuration and data
    print("\n1. Loading data...")
    config = load_config()
    loader = DataLoader(
        data_path=config['data']['training_data'],
        target_column=config['data']['target_column']
    )
    loader.load_data()
    X, y = loader.split_features_target()
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Get sampler
    sampling_config = config.get('sampling', {})
    sampler = get_sampler(
        method=sampling_config.get('method', 'none'),
        k_neighbors=sampling_config.get('k_neighbors', 5),
        sampling_strategy=sampling_config.get('sampling_strategy', 'auto'),
        random_state=sampling_config.get('random_state', 42)
    )
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = Preprocessor(
        scale_features=True,
        remove_constant_features=True,
        remove_redundant_features=True,
        sampler=sampler
    )
    
    # Fit preprocessor
    X_scaled = preprocessor.fit_transform(X)
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    # Apply sampling if configured
    if sampler is not None:
        print(f"\n3. Applying {sampler.__class__.__name__}...")
        X_scaled, y_arr = sampler.fit_resample(X_scaled, y_arr)
        print(f"   Dataset size after sampling: {len(X_scaled)}")
    
    print(f"   Final feature count: {X_scaled.shape[1]}")
    
    # Define search space
    print(f"\n4. Defining search space...")
    search_space = {
        'model__hidden_layer_1': Integer(128, 512),
        'model__hidden_layer_2': Integer(64, 256),
        'model__hidden_layer_3': Integer(32, 128),
        'model__hidden_layer_4': Integer(0, 64),  # 0 means disabled
        'model__hidden_layer_5': Integer(0, 32),  # 0 means disabled
        'model__dropout_rate': Real(0.1, 0.5),
        'model__learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
        'model__optimizer': Categorical(['adam', 'adamw', 'rmsprop']),
        'batch_size': Categorical([16, 32, 64]),
        'epochs': Categorical([50, 100, 150])
    }
    
    print("   Search space:")
    for param, space in search_space.items():
        print(f"   - {param}: {space}")
    
    # Create model wrapper
    model = KerasClassifier(
        model=create_model_for_bayes,
        verbose=0,
        epochs=100,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )
        ],
        validation_split=0.2
    )
    
    # Setup Bayesian search
    print(f"\n5. Running Bayesian optimization ({n_iter} iterations)...")
    print(f"   Cross-validation: {cv_folds}-fold StratifiedKFold")
    print(f"   This may take 30-60 minutes...\n")
    
    # Define scoring to minimize misclassification error = 1 - accuracy
    misclf_error = make_scorer(lambda y_true, y_pred: 1 - accuracy_score(y_true, y_pred), greater_is_better=False)

    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        n_jobs=n_jobs,
        verbose=2,
        scoring=misclf_error,
        random_state=random_state,
        return_train_score=True
    )
    
    # Fit
    start_time = datetime.now()
    bayes_search.fit(X_scaled, y_arr)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\nBest parameters found:")
    for param, value in bayes_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation score: {bayes_search.best_score_:.4f}")
    print(f"Total optimization time: {duration:.1f} minutes")
    
    # Get top 5 configurations
    results_df = pd.DataFrame(bayes_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['mean_test_score', 'std_test_score', 'rank_test_score']
    ]
    
    print("\nTop 5 configurations:")
    print(top_5.to_string())
    
    # Save results
    print("\n6. Saving results...")
    results_dict = {
        'best_params': bayes_search.best_params_,
        'best_score': float(bayes_search.best_score_),
        'optimization_time_minutes': duration,
        'n_iterations': n_iter,
        'cv_folds': cv_folds,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'top_5_scores': top_5['mean_test_score'].tolist()
    }
    
    with open('results/logs/bayesian_optimization.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("   Results saved to: results/logs/bayesian_optimization.json")
    
    # Save full results
    results_df.to_csv('results/logs/bayesian_optimization_full.csv', index=False)
    print("   Full results saved to: results/logs/bayesian_optimization_full.csv")
    
    # Plot results
    plot_optimization_results(bayes_search)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    return bayes_search


if __name__ == "__main__":
    # Run optimization
    search = run_bayesian_optimization(
        n_iter=30,      # Number of optimization iterations
        cv_folds=5,     # Cross-validation folds
        n_jobs=1,       # Parallel jobs (use -1 for all cores)
        random_state=42
    )
    
    print("\nTo use the best parameters, update config.yaml with:")
    print("```yaml")
    print("model:")
    if 'model__hidden_layer_1' in search.best_params_:
        layers = [
            search.best_params_.get('model__hidden_layer_1', 128),
            search.best_params_.get('model__hidden_layer_2', 64),
            search.best_params_.get('model__hidden_layer_3', 32)
        ]
        # Add optional layers if they were used (non-zero)
        if search.best_params_.get('model__hidden_layer_4', 0) > 0:
            layers.append(search.best_params_['model__hidden_layer_4'])
        if search.best_params_.get('model__hidden_layer_5', 0) > 0:
            layers.append(search.best_params_['model__hidden_layer_5'])
        print(f"  hidden_layers: {layers}")
    if 'model__dropout_rate' in search.best_params_:
        print(f"  dropout_rate: {search.best_params_['model__dropout_rate']:.3f}")
    
    print("\ntraining:")
    if 'model__learning_rate' in search.best_params_:
        print(f"  learning_rate: {search.best_params_['model__learning_rate']:.6f}")
    if 'batch_size' in search.best_params_:
        print(f"  batch_size: {search.best_params_['batch_size']}")
    if 'epochs' in search.best_params_:
        print(f"  epochs: {search.best_params_['epochs']}")
    if 'model__optimizer' in search.best_params_:
        print(f"  optimizer: '{search.best_params_['model__optimizer']}'")
    print("```")

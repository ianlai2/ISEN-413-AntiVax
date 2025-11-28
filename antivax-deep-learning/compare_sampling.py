"""
Demonstrate different sampling methods for handling class imbalance.
Compare performance across different techniques.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.sampling import get_sampler
from src.models.model_builder import build_model_from_config
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model


def evaluate_sampling_method(
    X_train, X_test, y_train, y_test,
    config, method_name, sampler,
    seed=42
):
    """
    Evaluate a single sampling method.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        config: Configuration dict
        method_name: Name of sampling method
        sampler: Sampler instance or None
        seed: Random seed
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Testing: {method_name}")
    print(f"{'='*60}")
    
    set_seed(seed)
    
    # Apply sampling
    if sampler is not None:
        X_train_sampled, y_train_sampled = sampler.fit_resample(X_train, y_train)
    else:
        X_train_sampled = X_train
        y_train_sampled = y_train
        print("\nNo sampling applied (baseline)")
        print(f"  Training set size: {len(X_train)}")
    
    # Build and train model
    input_dim = X_train_sampled.shape[1]
    model = build_model_from_config(config, input_dim)
    
    trainer = Trainer(
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        validation_split=config['training']['validation_split'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        verbose=0
    )
    
    print("\nTraining model...")
    history = trainer.train(model, X_train_sampled, y_train_sampled)
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluate_model(model, X_test, y_test)
    
    print(f"\nResults for {method_name}:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    return {
        'method': method_name,
        'results': results,
        'history': history,
        'training_samples': len(X_train_sampled)
    }


def main():
    """Main comparison pipeline."""
    
    # Load configuration
    print("Loading configuration...")
    config = load_config('config/config.yaml')
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Load and preprocess data (without sampling)
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    data_path = config['data']['training_data']
    target_column = config['data']['target_column']
    
    loader = DataLoader(data_path, target_column)
    X, y = loader.split_features_target()
    
    # Preprocess without sampling
    preprocessor = Preprocessor(scale_features=True, sampler=None)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        X, y, test_size=0.2, random_state=seed, apply_sampling=False
    )
    
    # Define sampling methods to test
    sampling_methods = [
        ('No Sampling (Baseline)', None),
        ('Random Oversampling', get_sampler('oversample', random_state=seed)),
        ('Random Undersampling', get_sampler('undersample', random_state=seed)),
        ('SMOTE', get_sampler('smote', k_neighbors=5, random_state=seed)),
        ('SMOTE + Tomek', get_sampler('smote_tomek', k_neighbors=5, random_state=seed)),
        ('SMOTE + ENN', get_sampler('smote_enn', k_neighbors_smote=5, k_neighbors_enn=3, random_state=seed))
    ]
    
    # Evaluate each method
    all_results = []
    
    for method_name, sampler in sampling_methods:
        result = evaluate_sampling_method(
            X_train, X_test, y_train, y_test,
            config, method_name, sampler, seed
        )
        all_results.append(result)
    
    # Create comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    # Create DataFrame for comparison
    comparison_data = []
    for result in all_results:
        row = {
            'Method': result['method'],
            'Training Samples': result['training_samples']
        }
        row.update(result['results'])
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + df_comparison.to_string(index=False))
    
    # Save results
    results_dir = Path(config['paths']['logs'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df_comparison.to_csv(results_dir / 'sampling_comparison.csv', index=False)
    print(f"\nResults saved to: {results_dir / 'sampling_comparison.csv'}")
    
    # Create visualization
    print("\nGenerating comparison plots...")
    create_comparison_plots(df_comparison, config['paths']['plots'])
    
    # Print recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    best_accuracy = df_comparison.loc[df_comparison['accuracy'].idxmax()]
    best_auc = df_comparison.loc[df_comparison['auc'].idxmax()]
    best_f1 = df_comparison.loc[df_comparison['f1'].idxmax()]
    
    print(f"\nBest Accuracy: {best_accuracy['Method']} ({best_accuracy['accuracy']:.4f})")
    print(f"Best AUC: {best_auc['Method']} ({best_auc['auc']:.4f})")
    print(f"Best F1-Score: {best_f1['Method']} ({best_f1['f1']:.4f})")
    
    return all_results, df_comparison


def create_comparison_plots(df: pd.DataFrame, plots_dir: str):
    """
    Create comparison plots for different sampling methods.
    
    Args:
        df: Comparison DataFrame
        plots_dir: Directory to save plots
    """
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Create bar plot
        colors = ['#d62728' if i == 0 else '#2ca02c' for i in range(len(df))]
        bars = ax.bar(range(len(df)), df[metric], color=colors, alpha=0.7, edgecolor='black')
        
        # Customize
        ax.set_xlabel('Sampling Method', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()} by Sampling Method', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Method'], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Highlight best
        best_idx = df[metric].idxmax()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
    
    # Training samples plot
    ax = axes[5]
    bars = ax.bar(range(len(df)), df['Training Samples'], color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Sampling Method', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax.set_title('Training Set Size', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Method'], rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    save_path = plots_path / 'sampling_methods_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved comparison plot: {save_path}")
    
    plt.close()
    
    # Create radar chart for best methods
    create_radar_chart(df, plots_path)


def create_radar_chart(df: pd.DataFrame, plots_path: Path):
    """
    Create radar chart comparing top sampling methods.
    
    Args:
        df: Comparison DataFrame
        plots_path: Path to save plots
    """
    # Select metrics for radar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Normalize metrics to 0-1 scale for better visualization
    df_norm = df.copy()
    for metric in metrics:
        df_norm[metric] = (df_norm[metric] - df_norm[metric].min()) / (df_norm[metric].max() - df_norm[metric].min())
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot each method
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for idx, row in df_norm.iterrows():
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Method'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11, fontweight='bold')
    
    # Set title and legend
    ax.set_title('Sampling Methods Performance Comparison\n(Normalized Metrics)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    ax.grid(True)
    
    # Save
    save_path = plots_path / 'sampling_methods_radar.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved radar chart: {save_path}")
    
    plt.close()


if __name__ == '__main__':
    print("Sampling Methods Comparison Tool")
    print("This will test 6 different sampling approaches and compare their performance.\n")
    
    results, comparison_df = main()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

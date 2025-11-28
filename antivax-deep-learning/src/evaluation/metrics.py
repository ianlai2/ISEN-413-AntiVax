"""
Model evaluation metrics and visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from typing import Dict, Any, List, Optional
from tensorflow import keras
from pathlib import Path


def evaluate_model(model: keras.Model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance on given data.
    
    Args:
        model: Trained Keras model
        X: Features
        y: True targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    
    # Misclassification error: (1/n) * sum of indicator(y_pred != y_true)
    n_mislabeled = np.sum(y_pred != y)
    misclassification_error = n_mislabeled / len(y)
    
    metrics = {
        'accuracy': accuracy,
        'misclassification_error': misclassification_error,
        'n_mislabeled': int(n_mislabeled),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'auc': roc_auc_score(y, y_pred_proba)
    }
    
    return metrics


def detailed_evaluation(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    print_report: bool = True
) -> Dict[str, Any]:
    """
    Perform detailed model evaluation.
    
    Args:
        model: Trained Keras model
        X: Features
        y: True targets
        print_report: Whether to print classification report
        
    Returns:
        Dictionary containing metrics and confusion matrix
    """
    # Get predictions
    y_pred_proba = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = evaluate_model(model, X, y)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    results = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'y_true': y
    }
    
    if print_report:
        print("\n" + "="*60)
        print("Evaluation Report")
        print("="*60)
        print(f"\nAccuracy:                {metrics['accuracy']:.4f}")
        print(f"Misclassification Error:  {metrics['misclassification_error']:.4f}")
        print(f"Mislabeled Points:        {metrics['n_mislabeled']}/{len(y)}")
        print(f"Precision:               {metrics['precision']:.4f}")
        print(f"Recall:                  {metrics['recall']:.4f}")
        print(f"F1 Score:                {metrics['f1_score']:.4f}")
        print(f"AUC:                     {metrics['auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(y, y_pred, zero_division=0))
    
    return results


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                   auc_score: float, save_path: Optional[str] = None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        auc_score: AUC score
        save_path: Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_training_history(history: keras.callbacks.History, 
                          save_path: Optional[str] = None):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Keras training history
        save_path: Path to save the plot
    """
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # AUC
    axes[1, 0].plot(epochs, history_dict['auc'], 'b-', label='Training AUC', linewidth=2)
    axes[1, 0].plot(epochs, history_dict['val_auc'], 'r-', label='Validation AUC', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('AUC', fontsize=11)
    axes[1, 0].set_title('Training and Validation AUC', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    
    # Precision and Recall
    axes[1, 1].plot(epochs, history_dict['precision'], 'g-', label='Training Precision', linewidth=2)
    axes[1, 1].plot(epochs, history_dict['val_precision'], 'orange', label='Validation Precision', linewidth=2)
    axes[1, 1].plot(epochs, history_dict['recall'], 'b--', label='Training Recall', linewidth=1.5)
    axes[1, 1].plot(epochs, history_dict['val_recall'], 'r--', label='Validation Recall', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Score', fontsize=11)
    axes[1, 1].set_title('Precision and Recall', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    plt.close()


def plot_cv_results(cv_results: Dict, save_path: Optional[str] = None):
    """
    Plot cross-validation results showing metrics across folds.
    
    Args:
        cv_results: Cross-validation results dictionary
        save_path: Path to save the plot
    """
    aggregated = cv_results['aggregated_results']
    
    # Extract metrics for plotting
    metrics_to_plot = ['accuracy', 'misclassification_error', 'precision', 
                       'recall', 'f1_score', 'auc']
    metric_names = ['Accuracy', 'Misclass. Error', 'Precision', 
                    'Recall', 'F1-Score', 'AUC']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        if metric not in aggregated:
            continue
            
        values = aggregated[metric]['values']
        mean_val = aggregated[metric]['mean']
        std_val = aggregated[metric]['std']
        
        # Bar plot with error bars
        folds = [f'Fold {i+1}' for i in range(len(values))]
        axes[idx].bar(folds, values, alpha=0.7, color='skyblue', edgecolor='navy')
        axes[idx].axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
                         label=f'Mean = {mean_val:.4f}')
        axes[idx].axhline(y=mean_val + std_val, color='orange', linestyle=':', 
                         linewidth=1.5, alpha=0.7)
        axes[idx].axhline(y=mean_val - std_val, color='orange', linestyle=':', 
                         linewidth=1.5, alpha=0.7, label=f'Std = {std_val:.4f}')
        
        axes[idx].set_xlabel('Fold', fontsize=11)
        axes[idx].set_ylabel(name, fontsize=11)
        axes[idx].set_title(f'{name} Across Folds', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CV results plot saved to {save_path}")
    plt.close()


def plot_metrics_comparison(cv_metrics: Dict, test_metrics: Dict, 
                           save_path: Optional[str] = None):
    """
    Plot comparison between CV and test set metrics.
    
    Args:
        cv_metrics: Cross-validation aggregated metrics
        test_metrics: Test set metrics
        save_path: Path to save the plot
    """
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 
                          'misclassification_error']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 
                     'Misclass. Error']
    
    cv_values = [cv_metrics[m]['mean'] for m in metrics_to_compare if m in cv_metrics]
    cv_stds = [cv_metrics[m]['std'] for m in metrics_to_compare if m in cv_metrics]
    test_values = [test_metrics[m] for m in metrics_to_compare if m in test_metrics]
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, cv_values, width, yerr=cv_stds, 
                   label='CV Mean ± Std', alpha=0.8, color='steelblue', 
                   capsize=5, edgecolor='navy')
    bars2 = ax.bar(x + width/2, test_values, width, 
                   label='Test Set', alpha=0.8, color='coral', 
                   edgecolor='darkred')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation vs Test Set Performance', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=30, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    plt.close()


def generate_all_plots(cv_results: Dict, test_results: Dict, history: keras.callbacks.History,
                      plots_dir: str = "results/plots"):
    """
    Generate all diagnostic plots.
    
    Args:
        cv_results: Cross-validation results
        test_results: Test set results
        history: Training history
        plots_dir: Directory to save plots
    """
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Diagnostic Plots")
    print("="*60)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 1. Confusion Matrix
    plot_confusion_matrix(test_results['confusion_matrix'], 
                         str(plots_path / 'confusion_matrix.png'))
    
    # 2. ROC Curve
    plot_roc_curve(test_results['y_true'], test_results['probabilities'], 
                  test_results['metrics']['auc'],
                  str(plots_path / 'roc_curve.png'))
    
    # 3. Training History
    plot_training_history(history, str(plots_path / 'training_history.png'))
    
    # 4. Cross-Validation Results
    plot_cv_results(cv_results, str(plots_path / 'cv_results.png'))
    
    # 5. CV vs Test Comparison
    plot_metrics_comparison(cv_results['aggregated_results'], 
                           test_results['metrics'],
                           str(plots_path / 'cv_vs_test.png'))
    
    print("\n[SUCCESS] All diagnostic plots generated successfully!")
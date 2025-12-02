"""
Comprehensive Feature Selection

This module implements multiple feature selection methods to identify
the most important features for anti-vax prediction:
- ANOVA F-test (univariate statistical test)
- Mutual Information (information theory)
- Random Forest feature importance (tree-based)
- Recursive Feature Elimination with CV (wrapper method)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, RFECV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
import json

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.sampling import get_sampler


def plot_feature_selection_results(f_scores, mi_scores, rf_scores, rfecv, 
                                   consensus_features, save_path='results/plots/feature_selection.png'):
    """
    Visualize feature selection results.
    
    Args:
        f_scores: DataFrame with F-test scores
        mi_scores: DataFrame with mutual information scores
        rf_scores: DataFrame with Random Forest importances
        rfecv: Fitted RFECV object
        consensus_features: List of consensus features
        save_path: Path to save plot
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 1. ANOVA F-test top 20
    ax1 = plt.subplot(3, 3, 1)
    top_f = f_scores.head(20).sort_values('f_score')
    ax1.barh(range(len(top_f)), top_f['f_score'], color='steelblue')
    ax1.set_yticks(range(len(top_f)))
    ax1.set_yticklabels(top_f['feature'], fontsize=8)
    ax1.set_xlabel('F-Score', fontsize=10)
    ax1.set_title('Top 20 Features - ANOVA F-test', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Mutual Information top 20
    ax2 = plt.subplot(3, 3, 2)
    top_mi = mi_scores.head(20).sort_values('mi_score')
    ax2.barh(range(len(top_mi)), top_mi['mi_score'], color='darkorange')
    ax2.set_yticks(range(len(top_mi)))
    ax2.set_yticklabels(top_mi['feature'], fontsize=8)
    ax2.set_xlabel('MI Score', fontsize=10)
    ax2.set_title('Top 20 Features - Mutual Information', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Random Forest importance top 20
    ax3 = plt.subplot(3, 3, 3)
    top_rf = rf_scores.head(20).sort_values('importance')
    ax3.barh(range(len(top_rf)), top_rf['importance'], color='forestgreen')
    ax3.set_yticks(range(len(top_rf)))
    ax3.set_yticklabels(top_rf['feature'], fontsize=8)
    ax3.set_xlabel('Importance', fontsize=10)
    ax3.set_title('Top 20 Features - Random Forest', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. RFE CV curve
    ax4 = plt.subplot(3, 3, 4)
    n_features = len(rfecv.cv_results_['mean_test_score'])
    ax4.plot(range(1, n_features + 1), rfecv.cv_results_['mean_test_score'], 
             linewidth=2, marker='o', markersize=4, color='steelblue')
    ax4.axvline(rfecv.n_features_, color='red', linestyle='--', linewidth=2,
                label=f'Optimal: {rfecv.n_features_} features')
    ax4.fill_between(range(1, n_features + 1),
                      rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'],
                      rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'],
                      alpha=0.2)
    ax4.set_xlabel('Number of Features Selected', fontsize=10)
    ax4.set_ylabel('Cross-Validation Score', fontsize=10)
    ax4.set_title('RFE with Cross-Validation', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Method agreement heatmap
    ax5 = plt.subplot(3, 3, 5)
    
    # Get top 30 from each method
    top_f_set = set(f_scores.head(30)['feature'])
    top_mi_set = set(mi_scores.head(30)['feature'])
    top_rf_set = set(rf_scores.head(30)['feature'])
    consensus_set = set(consensus_features)
    
    # Create agreement matrix
    methods = ['F-test\n(top 30)', 'MI\n(top 30)', 'RF\n(top 30)', 'Consensus\n(3+ votes)']
    agreement = np.array([
        [len(top_f_set), len(top_f_set & top_mi_set), len(top_f_set & top_rf_set), len(top_f_set & consensus_set)],
        [len(top_mi_set & top_f_set), len(top_mi_set), len(top_mi_set & top_rf_set), len(top_mi_set & consensus_set)],
        [len(top_rf_set & top_f_set), len(top_rf_set & top_mi_set), len(top_rf_set), len(top_rf_set & consensus_set)],
        [len(consensus_set & top_f_set), len(consensus_set & top_mi_set), len(consensus_set & top_rf_set), len(consensus_set)]
    ])
    
    sns.heatmap(agreement, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=methods, yticklabels=methods, ax=ax5, cbar_kws={'label': 'Count'})
    ax5.set_title('Feature Selection Method Agreement', fontsize=12, fontweight='bold')
    
    # 6. Score distributions
    ax6 = plt.subplot(3, 3, 6)
    normalized_f = (f_scores['f_score'] - f_scores['f_score'].min()) / (f_scores['f_score'].max() - f_scores['f_score'].min())
    normalized_mi = (mi_scores['mi_score'] - mi_scores['mi_score'].min()) / (mi_scores['mi_score'].max() - mi_scores['mi_score'].min())
    normalized_rf = (rf_scores['importance'] - rf_scores['importance'].min()) / (rf_scores['importance'].max() - rf_scores['importance'].min())
    
    ax6.hist(normalized_f, bins=30, alpha=0.5, label='F-test', color='steelblue')
    ax6.hist(normalized_mi, bins=30, alpha=0.5, label='MI', color='darkorange')
    ax6.hist(normalized_rf, bins=30, alpha=0.5, label='RF', color='forestgreen')
    ax6.set_xlabel('Normalized Score', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Score Distributions (Normalized)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Consensus features (top 20)
    ax7 = plt.subplot(3, 3, 7)
    if len(consensus_features) > 0:
        # Count votes for each consensus feature
        consensus_data = []
        for feat in consensus_features[:20]:
            votes = 0
            if feat in top_f_set: votes += 1
            if feat in top_mi_set: votes += 1
            if feat in top_rf_set: votes += 1
            consensus_data.append((feat, votes))
        
        consensus_data.sort(key=lambda x: x[1])
        features, votes = zip(*consensus_data)
        
        colors = ['red' if v == 4 else 'orange' if v == 3 else 'steelblue' for v in votes]
        ax7.barh(range(len(features)), votes, color=colors)
        ax7.set_yticks(range(len(features)))
        ax7.set_yticklabels(features, fontsize=8)
        ax7.set_xlabel('Number of Methods', fontsize=10)
        ax7.set_title(f'Top {len(features)} Consensus Features', fontsize=12, fontweight='bold')
        ax7.set_xlim([0, 4.5])
        ax7.set_xticks([1, 2, 3, 4])
        ax7.grid(True, alpha=0.3, axis='x')
    
    # 8. Feature importance comparison (top 15)
    ax8 = plt.subplot(3, 3, 8)
    top_15_features = consensus_features[:15] if len(consensus_features) >= 15 else consensus_features
    
    if len(top_15_features) > 0:
        # Normalize scores for comparison
        f_dict = dict(zip(f_scores['feature'], normalized_f))
        mi_dict = dict(zip(mi_scores['feature'], normalized_mi))
        rf_dict = dict(zip(rf_scores['importance'].index, normalized_rf))
        
        x = np.arange(len(top_15_features))
        width = 0.25
        
        f_vals = [f_dict.get(f, 0) for f in top_15_features]
        mi_vals = [mi_dict.get(f, 0) for f in top_15_features]
        rf_vals = [rf_dict.get(f, 0) for f in top_15_features]
        
        ax8.bar(x - width, f_vals, width, label='F-test', color='steelblue', alpha=0.8)
        ax8.bar(x, mi_vals, width, label='MI', color='darkorange', alpha=0.8)
        ax8.bar(x + width, rf_vals, width, label='RF', color='forestgreen', alpha=0.8)
        
        ax8.set_xlabel('Features', fontsize=10)
        ax8.set_ylabel('Normalized Score', fontsize=10)
        ax8.set_title('Method Comparison (Top 15 Consensus)', fontsize=12, fontweight='bold')
        ax8.set_xticks(x)
        ax8.set_xticklabels(top_15_features, rotation=45, ha='right', fontsize=8)
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    FEATURE SELECTION SUMMARY
    
    Total Features: {len(f_scores)}
    
    Top Features per Method:
    • F-test (top 50): {len(f_scores.head(50))}
    • Mutual Info (top 50): {len(mi_scores.head(50))}
    • Random Forest (top 50): {len(rf_scores.head(50))}
    • RFE Optimal: {rfecv.n_features_}
    
    Consensus Features (3+ votes): {len(consensus_features)}
    
    Method Agreement:
    • F-test ∩ MI: {len(top_f_set & top_mi_set)}
    • F-test ∩ RF: {len(top_f_set & top_rf_set)}
    • MI ∩ RF: {len(top_mi_set & top_rf_set)}
    • All 3 methods: {len(top_f_set & top_mi_set & top_rf_set)}
    
    Recommendation:
    Use {len(consensus_features)} consensus features
    for improved model performance.
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFeature selection visualization saved to: {save_path}")
    plt.close()


def comprehensive_feature_selection(X_train, y_train, feature_names, top_k=50, cv_folds=5):
    """
    Run multiple feature selection methods and find consensus.
    
    Args:
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        feature_names: List of feature names
        top_k: Number of top features to select per method
        cv_folds: Number of CV folds for RFE
        
    Returns:
        Dictionary with results from each method and consensus features
    """
    results = {}
    
    print("\n" + "="*70)
    print("COMPREHENSIVE FEATURE SELECTION")
    print("="*70)
    
    # 1. ANOVA F-test
    print("\n1. Running ANOVA F-test...")
    selector_f = SelectKBest(f_classif, k='all')
    selector_f.fit(X_train, y_train)
    f_scores = pd.DataFrame({
        'feature': feature_names,
        'f_score': selector_f.scores_
    }).sort_values('f_score', ascending=False)
    results['f_test'] = f_scores.head(top_k)['feature'].tolist()
    print(f"   Top feature: {f_scores.iloc[0]['feature']} (score: {f_scores.iloc[0]['f_score']:.2f})")
    
    # 2. Mutual Information
    print("\n2. Running Mutual Information...")
    selector_mi = SelectKBest(mutual_info_classif, k='all')
    selector_mi.fit(X_train, y_train)
    mi_scores = pd.DataFrame({
        'feature': feature_names,
        'mi_score': selector_mi.scores_
    }).sort_values('mi_score', ascending=False)
    results['mutual_info'] = mi_scores.head(top_k)['feature'].tolist()
    print(f"   Top feature: {mi_scores.iloc[0]['feature']} (score: {mi_scores.iloc[0]['mi_score']:.4f})")
    
    # 3. Random Forest Feature Importance
    print("\n3. Running Random Forest importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_scores = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    results['random_forest'] = rf_scores.head(top_k)['feature'].tolist()
    print(f"   Top feature: {rf_scores.iloc[0]['feature']} (importance: {rf_scores.iloc[0]['importance']:.4f})")
    print(f"   Random Forest accuracy: {rf.score(X_train, y_train):.4f}")
    
    # 4. Recursive Feature Elimination with CV
    print(f"\n4. Running RFE with {cv_folds}-fold CV...")
    print("   This may take several minutes...")
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    rfecv = RFECV(estimator, step=1, cv=cv_folds, scoring='accuracy', n_jobs=-1)
    rfecv.fit(X_train, y_train)
    rfe_features = [f for f, s in zip(feature_names, rfecv.support_) if s]
    results['rfe'] = rfe_features
    print(f"   Optimal number of features: {rfecv.n_features_}")
    print(f"   Best CV score: {rfecv.cv_results_['mean_test_score'][rfecv.n_features_-1]:.4f}")
    
    # Find consensus features (appear in at least 3 methods)
    print("\n5. Finding consensus features...")
    all_features = (results['f_test'] + results['mutual_info'] + 
                   results['random_forest'] + results['rfe'])
    feature_counts = Counter(all_features)
    
    # Features appearing in 3+ methods
    consensus_3plus = [f for f, count in feature_counts.items() if count >= 3]
    # Features appearing in 2+ methods
    consensus_2plus = [f for f, count in feature_counts.items() if count >= 2]
    
    # Sort by count
    consensus_3plus.sort(key=lambda x: feature_counts[x], reverse=True)
    consensus_2plus.sort(key=lambda x: feature_counts[x], reverse=True)
    
    results['consensus_3plus'] = consensus_3plus
    results['consensus_2plus'] = consensus_2plus
    
    print(f"   Features appearing in 3+ methods: {len(consensus_3plus)}")
    print(f"   Features appearing in 2+ methods: {len(consensus_2plus)}")
    
    # Print consensus features
    print("\n" + "="*70)
    print("CONSENSUS FEATURES (3+ methods)")
    print("="*70)
    for feat in consensus_3plus[:20]:
        count = feature_counts[feat]
        methods = []
        if feat in results['f_test']: methods.append('F-test')
        if feat in results['mutual_info']: methods.append('MI')
        if feat in results['random_forest']: methods.append('RF')
        if feat in results['rfe']: methods.append('RFE')
        print(f"  {feat:20s} ({count}/4 methods: {', '.join(methods)})")
    
    if len(consensus_3plus) > 20:
        print(f"  ... and {len(consensus_3plus) - 20} more")
    
    # Visualize
    plot_feature_selection_results(f_scores, mi_scores, rf_scores, rfecv, 
                                   consensus_3plus)
    
    return results, f_scores, mi_scores, rf_scores, rfecv


def save_feature_selection_results(results, f_scores, mi_scores, rf_scores, rfecv):
    """Save feature selection results to files."""
    print("\n6. Saving results...")
    
    # Ensure JSON-serializable primitives (convert numpy types to Python types)
    def to_builtin(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    # Save summary
    summary = {
        'total_features': int(len(f_scores)),
        'f_test_top_50': results['f_test'],
        'mutual_info_top_50': results['mutual_info'],
        'random_forest_top_50': results['random_forest'],
        'rfe_optimal_count': int(to_builtin(rfecv.n_features_)),
        'rfe_selected_features': results['rfe'],
        'consensus_3plus': results['consensus_3plus'],
        'consensus_2plus': results['consensus_2plus'],
        'n_consensus_3plus': int(len(results['consensus_3plus'])),
        'n_consensus_2plus': int(len(results['consensus_2plus']))
    }
    
    with open('results/logs/feature_selection.json', 'w') as f:
        # Map through to_builtin to avoid numpy scalar serialization issues
        json.dump({k: to_builtin(v) for k, v in summary.items()}, f, indent=2)
    print("   Summary saved to: results/logs/feature_selection.json")
    
    # Save detailed scores
    all_scores = f_scores.copy()
    all_scores = all_scores.merge(mi_scores, on='feature', how='left')
    all_scores = all_scores.merge(rf_scores, on='feature', how='left')
    all_scores.to_csv('results/logs/feature_scores.csv', index=False)
    print("   Detailed scores saved to: results/logs/feature_scores.csv")
    
    # Save RFE CV results
    rfe_results = pd.DataFrame({
        'n_features': range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
        'mean_score': rfecv.cv_results_['mean_test_score'],
        'std_score': rfecv.cv_results_['std_test_score']
    })
    rfe_results.to_csv('results/logs/rfe_cv_results.csv', index=False)
    print("   RFE CV results saved to: results/logs/rfe_cv_results.csv")


if __name__ == "__main__":
    # Set seed
    set_seed(42)
    
    # Load configuration and data
    print("Loading data...")
    config = load_config()
    loader = DataLoader(
        data_path=config['data']['training_data'],
        target_column=config['data']['target_column']
    )
    loader.load_data()
    X, y = loader.split_features_target()
    
    print(f"Dataset shape: {X.shape}")
    
    # Get sampler
    sampling_config = config.get('sampling', {})
    sampler = get_sampler(
        method=sampling_config.get('method', 'none'),
        k_neighbors=sampling_config.get('k_neighbors', 5),
        sampling_strategy=sampling_config.get('sampling_strategy', 'auto'),
        random_state=sampling_config.get('random_state', 42)
    )
    
    # Preprocess
    print("\nPreprocessing data...")
    preprocessor = Preprocessor(
        scale_features=True,
        remove_constant_features=True,
        remove_redundant_features=True,
        sampler=sampler
    )
    
    X_scaled = preprocessor.fit_transform(X)
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    # Apply sampling
    if sampler is not None:
        print(f"\nApplying {sampler.__class__.__name__}...")
        X_scaled, y_arr = sampler.fit_resample(X_scaled, y_arr)
    
    # Get feature names after preprocessing
    feature_names = [col for col in X.columns if col not in preprocessor.features_to_remove]
    print(f"\nFinal feature count: {len(feature_names)}")
    
    # Run feature selection
    results, f_scores, mi_scores, rf_scores, rfecv = comprehensive_feature_selection(
        X_scaled, y_arr, feature_names, top_k=50, cv_folds=5
    )
    
    # Save results
    save_feature_selection_results(results, f_scores, mi_scores, rf_scores, rfecv)
    
    print("\n" + "="*70)
    print("FEATURE SELECTION COMPLETE")
    print("="*70)
    print(f"\nRecommendation: Use {len(results['consensus_3plus'])} consensus features")
    print("These features were selected by at least 3 different methods.")

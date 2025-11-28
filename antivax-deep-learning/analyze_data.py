"""
Data analysis script to identify categorical variables and redundant predictors.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.config import load_config


def analyze_data():
    """Analyze the training data for categorical variables and redundancy."""
    
    # Load configuration
    config = load_config('config/config.yaml')
    data_path = config['data']['training_data']
    target_column = config['data']['target_column']
    
    print("="*70)
    print("DATA ANALYSIS - Categorical Variables & Redundant Predictors")
    print("="*70)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1} (excluding target)")
    print(f"Samples: {df.shape[0]}")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"\n{'='*70}")
    print("CATEGORICAL VARIABLE ANALYSIS")
    print("="*70)
    
    # Identify potentially categorical variables
    categorical_candidates = []
    binary_vars = []
    
    for col in X.columns:
        unique_vals = X[col].nunique()
        unique_list = sorted(X[col].unique())
        
        # Check for binary variables (0/1 or similar)
        if unique_vals == 2:
            binary_vars.append({
                'column': col,
                'unique_values': unique_list,
                'counts': X[col].value_counts().to_dict()
            })
        # Check for low cardinality (likely categorical)
        elif unique_vals <= 10:
            categorical_candidates.append({
                'column': col,
                'unique_count': unique_vals,
                'unique_values': unique_list,
                'counts': X[col].value_counts().to_dict()
            })
    
    print(f"\nBinary Variables (2 unique values): {len(binary_vars)}")
    if binary_vars:
        for var in binary_vars[:10]:  # Show first 10
            print(f"  - {var['column']}: {var['unique_values']} | Counts: {var['counts']}")
        if len(binary_vars) > 10:
            print(f"  ... and {len(binary_vars) - 10} more")
    
    print(f"\nLow Cardinality Variables (3-10 unique values): {len(categorical_candidates)}")
    if categorical_candidates:
        for var in categorical_candidates:
            print(f"  - {var['column']}: {var['unique_count']} unique | Values: {var['unique_values']}")
    
    print(f"\n{'='*70}")
    print("REDUNDANT PREDICTOR ANALYSIS")
    print("="*70)
    
    # Check for missing indicator columns
    missing_indicators = [col for col in X.columns if col.endswith('_missing')]
    print(f"\nMissing Indicator Columns: {len(missing_indicators)}")
    if missing_indicators:
        for col in missing_indicators:
            base_col = col.replace('_missing', '')
            if base_col in X.columns:
                # Check if missing indicator is all zeros (redundant)
                if X[col].sum() == 0:
                    print(f"  - {col}: REDUNDANT (all zeros, no missing values)")
                else:
                    missing_count = X[col].sum()
                    missing_pct = (missing_count / len(X)) * 100
                    print(f"  - {col}: {missing_count} missing ({missing_pct:.2f}%)")
    
    # Check for constant columns
    print(f"\nConstant Columns (zero variance):")
    constant_cols = []
    for col in X.columns:
        if X[col].nunique() == 1:
            constant_cols.append(col)
            print(f"  - {col}: constant value = {X[col].unique()[0]}")
    
    if not constant_cols:
        print("  None found")
    
    # Check for highly correlated features
    print(f"\nHighly Correlated Features (|correlation| > 0.95):")
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = []
    for column in upper_triangle.columns:
        high_corr = upper_triangle[column][upper_triangle[column] > 0.95]
        for idx in high_corr.index:
            high_corr_pairs.append({
                'feature1': column,
                'feature2': idx,
                'correlation': corr_matrix.loc[idx, column]
            })
    
    if high_corr_pairs:
        for pair in high_corr_pairs[:20]:  # Show first 20
            print(f"  - {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")
        if len(high_corr_pairs) > 20:
            print(f"  ... and {len(high_corr_pairs) - 20} more pairs")
    else:
        print("  None found")
    
    # Generate summary report
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("="*70)
    
    redundant_features = []
    
    # Redundant missing indicators (all zeros)
    for col in missing_indicators:
        if X[col].sum() == 0:
            redundant_features.append(col)
    
    # Constant columns
    redundant_features.extend(constant_cols)
    
    print(f"\nFeatures to Remove ({len(redundant_features)}):")
    if redundant_features:
        for feat in redundant_features:
            print(f"  - {feat}")
    else:
        print("  None identified")
    
    print(f"\nBinary Variables: Already suitable for neural networks (0/1 encoding)")
    print(f"Low Cardinality Variables: Consider if they represent ordered categories")
    
    # Save analysis results
    results_path = Path('results/logs')
    results_path.mkdir(parents=True, exist_ok=True)
    
    with open(results_path / 'data_analysis.txt', 'w') as f:
        f.write("DATA ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Features: {X.shape[1]}\n")
        f.write(f"Binary Variables: {len(binary_vars)}\n")
        f.write(f"Low Cardinality Variables: {len(categorical_candidates)}\n")
        f.write(f"Missing Indicators: {len(missing_indicators)}\n")
        f.write(f"Redundant Features: {len(redundant_features)}\n\n")
        f.write("Redundant Features to Remove:\n")
        for feat in redundant_features:
            f.write(f"  - {feat}\n")
    
    print(f"\nAnalysis saved to: {results_path / 'data_analysis.txt'}")
    
    return {
        'binary_vars': binary_vars,
        'categorical_candidates': categorical_candidates,
        'missing_indicators': missing_indicators,
        'redundant_features': redundant_features,
        'high_corr_pairs': high_corr_pairs
    }


if __name__ == '__main__':
    results = analyze_data()

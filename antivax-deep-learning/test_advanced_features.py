"""
Quick Test - Verify Advanced Features Implementation

This script tests that all new modules can be imported and basic functionality works.
"""
import sys

print("="*70)
print("TESTING ADVANCED FEATURES IMPLEMENTATION")
print("="*70)

# Test 1: Import all new modules
print("\n1. Testing imports...")
try:
    from src.data.feature_engineering import FeatureEngineer
    print("   ✓ FeatureEngineer imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import FeatureEngineer: {e}")
    sys.exit(1)

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    print("   ✓ scikit-optimize imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import scikit-optimize: {e}")
    sys.exit(1)

try:
    from scikeras.wrappers import KerasClassifier
    print("   ✓ scikeras imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import scikeras: {e}")
    sys.exit(1)

try:
    from xgboost import XGBClassifier
    print("   ✓ xgboost imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import xgboost: {e}")
    sys.exit(1)

# Test 2: Load data and test feature engineering
print("\n2. Testing feature engineering...")
try:
    import numpy as np
    import pandas as pd
    from src.utils.config import load_config
    from src.data.data_loader import DataLoader
    
    config = load_config()
    loader = DataLoader(
        data_path=config['data']['training_data'],
        target_column=config['data']['target_column']
    )
    loader.load_data()
    X, y = loader.split_features_target()
    
    print(f"   ✓ Data loaded: {X.shape}")
    
    # Test feature engineering
    engineer = FeatureEngineer(
        enable_domain_features=True,
        enable_statistical_features=True,
        enable_interaction_features=True,
        enable_count_features=True
    )
    
    X_engineered = engineer.fit_transform(X)
    print(f"   ✓ Feature engineering successful")
    print(f"     Original features: {X.shape[1]}")
    print(f"     Engineered features: {X_engineered.shape[1]}")
    print(f"     New features created: {X_engineered.shape[1] - X.shape[1]}")
    
except Exception as e:
    print(f"   ✗ Feature engineering test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test preprocessor with feature engineering
print("\n3. Testing preprocessor integration...")
try:
    from src.data.preprocessor import Preprocessor
    from src.data.sampling import get_sampler
    
    sampler = get_sampler(
        method=config['sampling']['method'],
        k_neighbors=config['sampling'].get('k_neighbors', 5),
        sampling_strategy=config['sampling'].get('sampling_strategy', 'auto'),
        random_state=config['sampling'].get('random_state', 42)
    )
    
    # Create preprocessor with feature engineering
    engineer = FeatureEngineer(
        enable_domain_features=True,
        enable_statistical_features=False,
        enable_interaction_features=False,
        enable_count_features=False
    )
    
    preprocessor = Preprocessor(
        scale_features=True,
        remove_constant_features=True,
        remove_redundant_features=True,
        sampler=sampler,
        feature_engineer=engineer
    )
    
    X_processed = preprocessor.fit_transform(X)
    print(f"   ✓ Preprocessor with feature engineering successful")
    print(f"     Final processed features: {X_processed.shape[1]}")
    
except Exception as e:
    print(f"   ✗ Preprocessor integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test feature selection functions
print("\n4. Testing feature selection imports...")
try:
    from sklearn.feature_selection import (
        SelectKBest, f_classif, mutual_info_classif,
        RFE, RFECV
    )
    from sklearn.ensemble import RandomForestClassifier
    print("   ✓ Feature selection modules imported successfully")
    
except Exception as e:
    print(f"   ✗ Feature selection import failed: {e}")
    sys.exit(1)

# Test 5: Verify config updates
print("\n5. Testing config updates...")
try:
    config = load_config()
    
    # Check feature engineering config exists
    if 'feature_engineering' in config:
        print("   ✓ feature_engineering section found in config")
        fe_config = config['feature_engineering']
        print(f"     - enable: {fe_config.get('enable', False)}")
        print(f"     - domain_features: {fe_config.get('domain_features', False)}")
        print(f"     - statistical_features: {fe_config.get('statistical_features', False)}")
        print(f"     - interaction_features: {fe_config.get('interaction_features', False)}")
        print(f"     - count_features: {fe_config.get('count_features', False)}")
    else:
        print("   ✗ feature_engineering section not found in config")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Config test failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nAdvanced features are ready to use!")
print("\nNext steps:")
print("  1. Run feature selection: python feature_selection.py")
print("  2. Run Bayesian optimization: python bayesian_optimization.py")
print("  3. Enable feature engineering in config.yaml")
print("  4. Train with optimal settings: python main.py")
print("\nSee ADVANCED_FEATURES_GUIDE.md for detailed instructions.")

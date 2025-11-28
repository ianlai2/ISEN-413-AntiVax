"""
Quick test of sampling methods to verify implementation.
"""
import numpy as np
from src.data.sampling import (
    get_sampler,
    RandomOverSampler,
    RandomUnderSampler,
    SMOTE,
    SMOTETomek,
    SMOTEENN
)


def create_imbalanced_data():
    """Create simple imbalanced dataset for testing."""
    np.random.seed(42)
    
    # Majority class (70%)
    X_majority = np.random.randn(700, 10)
    y_majority = np.zeros(700)
    
    # Minority class (30%)
    X_minority = np.random.randn(300, 10) + 1  # Shifted slightly
    y_minority = np.ones(300)
    
    # Combine
    X = np.vstack([X_majority, X_minority])
    y = np.concatenate([y_majority, y_minority])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


def test_sampler(sampler_name, sampler, X, y):
    """Test a single sampler."""
    print(f"\n{'='*60}")
    print(f"Testing: {sampler_name}")
    print(f"{'='*60}")
    
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Check results
        unique, counts = np.unique(y_resampled, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        print(f"\n✓ Success!")
        print(f"  Shape: {X.shape} → {X_resampled.shape}")
        print(f"  Distribution: {distribution}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("SAMPLING METHODS TEST SUITE")
    print("="*60)
    
    # Create test data
    print("\nCreating imbalanced test dataset...")
    X, y = create_imbalanced_data()
    
    unique, counts = np.unique(y, return_counts=True)
    print(f"Original distribution: {dict(zip(unique, counts))}")
    print(f"Shape: {X.shape}")
    
    # Test all samplers
    results = {}
    
    # 1. Random Oversampling
    sampler = RandomOverSampler(random_state=42)
    results['RandomOverSampler'] = test_sampler('RandomOverSampler', sampler, X, y)
    
    # 2. Random Undersampling
    sampler = RandomUnderSampler(random_state=42)
    results['RandomUnderSampler'] = test_sampler('RandomUnderSampler', sampler, X, y)
    
    # 3. SMOTE
    sampler = SMOTE(k_neighbors=5, random_state=42)
    results['SMOTE'] = test_sampler('SMOTE', sampler, X, y)
    
    # 4. SMOTE + Tomek
    sampler = SMOTETomek(k_neighbors=5, random_state=42)
    results['SMOTETomek'] = test_sampler('SMOTE + Tomek', sampler, X, y)
    
    # 5. SMOTE + ENN
    sampler = SMOTEENN(k_neighbors_smote=5, k_neighbors_enn=3, random_state=42)
    results['SMOTEENN'] = test_sampler('SMOTE + ENN', sampler, X, y)
    
    # 6. Test factory function
    print(f"\n{'='*60}")
    print("Testing Factory Function: get_sampler()")
    print(f"{'='*60}")
    
    methods = ['oversample', 'undersample', 'smote', 'smote_tomek', 'smote_enn']
    for method in methods:
        try:
            sampler = get_sampler(method, random_state=42)
            print(f"  ✓ get_sampler('{method}') → {sampler.__class__.__name__}")
        except Exception as e:
            print(f"  ✗ get_sampler('{method}') failed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    if passed == total:
        print(f"\n🎉 All tests passed! Sampling implementation is working correctly.")
    else:
        print(f"\n⚠ Some tests failed. Please review the output above.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

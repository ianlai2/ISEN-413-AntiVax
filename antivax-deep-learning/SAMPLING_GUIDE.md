# Class Imbalance Handling - Quick Guide

## What is Class Imbalance?

Class imbalance occurs when one class has significantly more samples than another. Example:
- Class 0 (Anti-vax): 70% of data
- Class 1 (Pro-vax): 30% of data

**Problem**: Models tend to favor the majority class, leading to poor minority class predictions.

---

## Quick Start

### Check Your Class Distribution

```python
from src.data.data_loader import DataLoader

loader = DataLoader("../fully_imputed_training_data.csv", "Y")
X, y = loader.split_features_target()

# This prints class distribution automatically
```

### Enable Sampling

Edit `config/config.yaml`:

```yaml
sampling:
  method: "smote"  # Change from 'none' to your chosen method
  k_neighbors: 5
  sampling_strategy: "auto"
  random_state: 42
```

Then run normally:
```bash
python main.py
```

---

## Available Methods

### 1. No Sampling (Default)
```yaml
sampling:
  method: "none"
```
- **Use when**: Classes are balanced (40-60% split)
- **Speed**: N/A
- **Effect**: No changes to data

### 2. Random Oversampling
```yaml
sampling:
  method: "oversample"
  sampling_strategy: "auto"  # Balance to majority class
```
- **Use when**: Need quick solution, small dataset
- **Speed**: ★★★★★ (Very fast)
- **Effect**: Duplicates minority class samples
- **Pros**: Simple, fast, no data loss
- **Cons**: Overfitting risk (exact duplicates)

**Example Result**:
```
Before: Class 0: 2800, Class 1: 1200
After:  Class 0: 2800, Class 1: 2800 (+1600 duplicates)
```

### 3. Random Undersampling
```yaml
sampling:
  method: "undersample"
  sampling_strategy: "auto"  # Balance to minority class
```
- **Use when**: Large dataset, data loss acceptable
- **Speed**: ★★★★★ (Very fast)
- **Effect**: Removes majority class samples
- **Pros**: Fast, reduces training time
- **Cons**: Loses potentially important data

**Example Result**:
```
Before: Class 0: 2800, Class 1: 1200
After:  Class 0: 1200, Class 1: 1200 (-1600 samples)
```

### 4. SMOTE (Recommended) ⭐
```yaml
sampling:
  method: "smote"
  k_neighbors: 5
  sampling_strategy: "auto"
```
- **Use when**: General purpose, most cases
- **Speed**: ★★★★ (Fast)
- **Effect**: Generates synthetic minority samples
- **Pros**: New data, reduces overfitting, industry standard
- **Cons**: Slightly slower, may create outliers

**How it works**:
1. Finds k nearest minority neighbors for each minority sample
2. Creates new sample between original and random neighbor
3. New sample = original + random_factor × (neighbor - original)

**Example Result**:
```
Before: Class 0: 2800, Class 1: 1200
After:  Class 0: 2800, Class 1: 2800 (+1600 synthetic)
```

### 5. SMOTE + Tomek Links
```yaml
sampling:
  method: "smote_tomek"
  k_neighbors: 5
  sampling_strategy: "auto"
```
- **Use when**: Classes overlap at boundaries, noisy data
- **Speed**: ★★★ (Moderate)
- **Effect**: SMOTE + removes borderline samples
- **Pros**: Cleaner decision boundary
- **Cons**: Slower, removes some data

**What are Tomek Links?**
- Pairs of samples from different classes that are nearest neighbors
- Indicates uncertain/ambiguous region
- Removing them clarifies the boundary

**Example Result**:
```
Before:        Class 0: 2800, Class 1: 1200
After SMOTE:   Class 0: 2800, Class 1: 2800
After Tomek:   Class 0: 2780, Class 1: 2800 (-20 borderline)
```

### 6. SMOTE + ENN
```yaml
sampling:
  method: "smote_enn"
  k_neighbors: 5
  k_neighbors_enn: 3
  sampling_strategy: "auto"
```
- **Use when**: Very noisy data, outliers present
- **Speed**: ★★ (Slower)
- **Effect**: SMOTE + removes samples with mismatched neighbors
- **Pros**: Most aggressive cleaning, best generalization
- **Cons**: Slowest, removes more data

**What is ENN (Edited Nearest Neighbors)?**
- Checks each sample's k nearest neighbors
- If sample's class ≠ majority of neighbors → remove (likely noise/outlier)

**Example Result**:
```
Before:        Class 0: 2800, Class 1: 1200
After SMOTE:   Class 0: 2800, Class 1: 2800
After ENN:     Class 0: 2750, Class 1: 2775 (-75 noisy)
```

---

## Quick Decision Tree

```
START: Is your data imbalanced?
│
├─ NO (40-60% split)
│  └─ Use: method: "none" ✓
│
└─ YES (< 30% minority)
   │
   ├─ Is your dataset small (< 2000)?
   │  ├─ YES → Use: "smote" (best for small data)
   │  └─ NO  → Continue
   │
   ├─ Is your dataset large (> 10000)?
   │  └─ YES → Try: "undersample" (fast) or "smote"
   │
   ├─ Is your data noisy/messy?
   │  ├─ YES → Use: "smote_enn" or "smote_tomek"
   │  └─ NO  → Use: "smote" ✓
   │
   └─ Do you have time constraints?
       ├─ YES → Use: "oversample" (fastest)
       └─ NO  → Use: "smote" (best quality) ✓
```

---

## Testing Different Methods

### Compare All Methods

Run the comparison script:
```bash
python compare_sampling.py
```

This will:
1. Test all 6 methods (none, oversample, undersample, SMOTE, SMOTE+Tomek, SMOTE+ENN)
2. Train a model with each
3. Compare accuracy, precision, recall, F1, AUC
4. Generate comparison plots
5. Save results to `results/logs/sampling_comparison.csv`

**Output**: 
- Console: Performance table
- File: `results/logs/sampling_comparison.csv`
- Plots: `results/plots/sampling_methods_comparison.png`
- Radar: `results/plots/sampling_methods_radar.png`

---

## Common Scenarios

### Scenario 1: Mild Imbalance (60-40)
**Recommendation**: No sampling needed
```yaml
sampling:
  method: "none"
```

### Scenario 2: Moderate Imbalance (70-30)
**Recommendation**: SMOTE
```yaml
sampling:
  method: "smote"
  k_neighbors: 5
```

### Scenario 3: Severe Imbalance (90-10)
**Recommendation**: SMOTE with more aggressive settings
```yaml
sampling:
  method: "smote"
  k_neighbors: 5
  sampling_strategy: "auto"
```
Or try SMOTE+ENN for cleaning:
```yaml
sampling:
  method: "smote_enn"
  k_neighbors: 5
  k_neighbors_enn: 3
```

### Scenario 4: Very Noisy Data
**Recommendation**: SMOTE + ENN
```yaml
sampling:
  method: "smote_enn"
  k_neighbors: 5
  k_neighbors_enn: 3
```

### Scenario 5: Large Dataset (>20k samples)
**Recommendation**: Random undersampling (fast) or SMOTE
```yaml
sampling:
  method: "undersample"  # Faster
  # OR
  method: "smote"        # Better quality
```

---

## Expected Improvements

### Before Sampling (Imbalanced)
```
Accuracy:  85%  ← Misleading (biased to majority)
Precision: 90%  ← High for majority
Recall:    60%  ← Low for minority
F1-Score:  72%  ← Poor minority performance
```

### After SMOTE
```
Accuracy:  83%  ← Slight drop (more balanced)
Precision: 84%  ← More balanced
Recall:    82%  ← Much better minority recall ✓
F1-Score:  83%  ← Better overall ✓
```

**Key**: Don't just look at accuracy! Check recall and F1 for minority class.

---

## Troubleshooting

### Problem: Sampling makes performance worse

**Possible Causes**:
1. Data wasn't actually imbalanced
2. Generated samples are too noisy
3. k_neighbors too small/large

**Solutions**:
- Check original class distribution
- Try different method (e.g., SMOTE → SMOTE+ENN)
- Adjust k_neighbors (try 3, 5, 7)

### Problem: Training takes too long after sampling

**Cause**: Dataset size increased significantly

**Solutions**:
1. Use undersampling instead
2. Use smaller batch size
3. Reduce model complexity
4. Use sampling_strategy: "minority" (less aggressive)

### Problem: Recall improves but precision drops

**Cause**: Too much oversampling, created too many synthetic samples

**Solutions**:
1. Use sampling_strategy: "minority" instead of "auto"
2. Try SMOTE+Tomek to clean boundaries
3. Reduce k_neighbors slightly

---

## Parameter Tuning

### k_neighbors (for SMOTE)

**Effect**: Number of neighbors used to generate synthetic samples

```yaml
k_neighbors: 3   # More variation, may be noisy
k_neighbors: 5   # Balanced (recommended) ✓
k_neighbors: 7   # More conservative, smoother
k_neighbors: 10  # Very conservative
```

**Rule of Thumb**: 
- Small minority class (<100) → k=3
- Medium minority class (100-500) → k=5 ✓
- Large minority class (>500) → k=7-10

### sampling_strategy

**Effect**: How to balance classes

```yaml
sampling_strategy: "auto"      # Balance minority to majority count
sampling_strategy: "minority"  # Balance to mean of both
```

**Example**:
```
Original: Class 0: 700, Class 1: 300

"auto":     Class 0: 700, Class 1: 700 (minority → majority)
"minority": Class 0: 700, Class 1: 500 (minority → mean)
```

---

## Best Practices

1. **Always split first, then sample**
   - ✓ Split into train/test
   - ✓ Apply sampling to training only
   - ✗ Never sample before splitting (data leakage!)

2. **Check class distribution**
   ```python
   print(y.value_counts(normalize=True))
   ```

3. **Start with SMOTE**
   - Most versatile
   - Good default choice
   - Then try others if needed

4. **Compare methods**
   ```bash
   python compare_sampling.py
   ```

5. **Monitor minority class metrics**
   - Focus on recall for minority class
   - Check F1-score for balance
   - Don't just trust accuracy

6. **Use cross-validation**
   - Sampling applied within each fold
   - More reliable performance estimate

---

## Summary Table

| Method | Speed | Quality | Data Loss | Best Use Case |
|--------|-------|---------|-----------|---------------|
| None | - | - | - | Balanced data |
| Oversample | ★★★★★ | ★★ | None | Quick tests |
| Undersample | ★★★★★ | ★★★ | High | Large datasets |
| **SMOTE** | ★★★★ | ★★★★★ | None | **General use ✓** |
| SMOTE+Tomek | ★★★ | ★★★★★ | Minimal | Noisy boundaries |
| SMOTE+ENN | ★★ | ★★★★★ | Some | Very noisy data |

---

## References

- Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
- Batista, G. E., Prati, R. C., & Monard, M. C. (2004). "A study of the behavior of several methods for balancing machine learning training data"

---

**Need Help?** See `PARAMETER_TUNING.md` for detailed explanations.

**Quick Test**: Run `python compare_sampling.py` to find the best method for your data.

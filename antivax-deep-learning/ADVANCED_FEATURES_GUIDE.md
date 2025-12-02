# Advanced Features: Bayesian Optimization & Feature Engineering

This guide explains the new advanced features for improving model performance.

## 🎯 Overview

Three new powerful tools have been implemented:

1. **Bayesian Optimization** - Efficiently find optimal hyperparameters
2. **Feature Selection** - Identify the most important features
3. **Feature Engineering** - Create new features from existing ones

---

## 1️⃣ Bayesian Optimization

### What It Does
Uses probabilistic models to intelligently search for the best hyperparameters. Much more efficient than grid search - finds better results with fewer iterations.

### Usage

```bash
python bayesian_optimization.py
```

### What It Optimizes
- Hidden layer sizes (64-256 neurons per layer)
- Dropout rate (0.1-0.5)
- Learning rate (0.0001-0.01)
- Optimizer (Adam, AdamW, RMSprop)
- Batch size (16, 32, 64)
- Epochs (50, 100, 150)

### Output
- `results/logs/bayesian_optimization.json` - Best parameters found
- `results/logs/bayesian_optimization_full.csv` - All tried configurations
- `results/plots/bayesian_optimization.png` - Visualization of optimization process

### Expected Runtime
30-60 minutes for 30 iterations (default)

### Example Output
```
Best parameters found:
  model__hidden_layer_1: 192
  model__hidden_layer_2: 96
  model__hidden_layer_3: 48
  model__dropout_rate: 0.35
  model__learning_rate: 0.000732
  model__optimizer: adamw
  batch_size: 32
  epochs: 100

Best cross-validation score: 0.8845
```

### How to Apply Results
After optimization completes, update `config/config.yaml`:

```yaml
model:
  hidden_layers: [192, 96, 48]
  dropout_rate: 0.35

training:
  learning_rate: 0.000732
  batch_size: 32
  epochs: 100
  optimizer: adamw
```

---

## 2️⃣ Feature Selection

### What It Does
Uses 4 different methods to identify the most important features:
- **ANOVA F-test** - Statistical significance
- **Mutual Information** - Information gain
- **Random Forest Importance** - Tree-based importance
- **Recursive Feature Elimination (RFE)** - Wrapper method with CV

### Usage

```bash
python feature_selection.py
```

### Output
- `results/logs/feature_selection.json` - Selected features summary
- `results/logs/feature_scores.csv` - Detailed scores for all features
- `results/logs/rfe_cv_results.csv` - RFE cross-validation curve
- `results/plots/feature_selection.png` - Comprehensive visualization

### Expected Runtime
5-15 minutes

### Key Concepts

**Consensus Features**: Features selected by 3+ methods
- These are the most reliable predictors
- Reduces features from ~123 to typically 40-60
- Improves model generalization

### Example Output
```
Consensus features (appear in 3+ methods): 47

Top consensus features:
  Q41_3      (4/4 methods: F-test, MI, RF, RFE)
  Q42_5      (4/4 methods: F-test, MI, RF, RFE)
  Q43_12     (4/4 methods: F-test, MI, RF, RFE)
  Q41_7      (3/4 methods: F-test, MI, RF)
  Q210       (3/4 methods: F-test, RF, RFE)
  ...
```

### How to Use Selected Features

The feature selection results are automatically saved. To use them with feature engineering:

1. Run feature selection first:
   ```bash
   python feature_selection.py
   ```

2. Enable in `config.yaml`:
   ```yaml
   feature_engineering:
     enable: true
     use_selected_features: true
     selected_features_file: "results/logs/feature_selection.json"
   ```

3. Run training:
   ```bash
   python main.py
   ```

---

## 3️⃣ Feature Engineering

### What It Does
Creates new features from existing ones to capture patterns the model might miss:

#### Domain Features
- Aggregations of survey question groups (Q41_*, Q42_*, etc.)
- Mean, std, min, max, range, median per question group
- Example: `Q41_mean`, `Q42_std`

#### Statistical Features
- Overall statistics across all features
- Mean, std, median, quartiles, IQR, CV
- Example: `overall_mean`, `overall_cv`

#### Interaction Features
- Ratios between feature groups
- Products of key features
- Differences from overall statistics
- Example: `Q41_Q42_ratio`, `Q41_vs_overall`

#### Count Features
- Count of specific values (1s, 2s, 3s, 4s, 5s)
- Count of zeros, max/min values
- Proportion above/below median
- Example: `count_3s`, `prop_above_median`

### Configuration

Edit `config/config.yaml`:

```yaml
feature_engineering:
  enable: true  # Enable feature engineering
  domain_features: true  # Question group aggregations
  statistical_features: true  # Overall statistics
  interaction_features: true  # Ratios and products
  count_features: true  # Value counts
  use_selected_features: false  # Use all features
```

### Usage

Feature engineering is integrated into the main pipeline:

```bash
python main.py
```

### Expected Impact
- Adds ~50-80 new features
- Can improve accuracy by 1-3%
- Better captures survey response patterns

### Example Output
```
Feature Engineering Summary:
  Original features: 123
  Engineered features: 187
  New features created: 64
```

---

## 🚀 Recommended Workflow

### Quick Win (30 minutes)
```bash
# 1. Run feature selection
python feature_selection.py

# 2. Train with selected features
# Edit config.yaml: set feature_engineering.use_selected_features = true
python main.py
```

**Expected gain**: +1-2% accuracy

---

### Full Optimization (2-3 hours)
```bash
# 1. Feature selection
python feature_selection.py

# 2. Enable feature engineering
# Edit config.yaml: 
#   feature_engineering.enable = true
#   feature_engineering.use_selected_features = true

# 3. Run Bayesian optimization with engineered features
python bayesian_optimization.py

# 4. Apply best parameters to config.yaml

# 5. Final training with optimal config
python main.py
```

**Expected gain**: +3-5% accuracy (88-90% total)

---

### Advanced Workflow (1 day)
```bash
# 1. Baseline with current config
python main.py

# 2. Feature selection
python feature_selection.py

# 3. Try feature engineering without selection
# config: enable=true, use_selected_features=false
python main.py

# 4. Try feature engineering with selection
# config: enable=true, use_selected_features=true
python main.py

# 5. Compare results, choose best approach

# 6. Run Bayesian optimization with best feature config
python bayesian_optimization.py

# 7. Apply optimal parameters
python main.py

# 8. Compare all sampling methods with optimal config
python compare_sampling.py
```

**Expected gain**: +4-6% accuracy (89-91% total)

---

## 📊 Performance Expectations

| Strategy | Time | Accuracy Gain | New Accuracy |
|----------|------|---------------|--------------|
| Baseline (current) | - | - | 86.5% |
| + Feature selection | 15 min | +1-2% | 87.5-88.5% |
| + Feature engineering | 5 min | +1-3% | 87.5-89.5% |
| + Bayesian optimization | 60 min | +2-4% | 88.5-90.5% |
| All combined | 90 min | +3-6% | 89.5-92.5% |

---

## 💡 Tips & Best Practices

### Feature Selection
- **Always run first** - Reduces noise and overfitting
- **Use consensus features** - More reliable than single method
- **Check RFE curve** - Shows optimal number of features
- **Save results** - Can be reused for future experiments

### Feature Engineering
- **Start simple** - Enable one type at a time to see impact
- **Monitor feature count** - Too many features can hurt performance
- **Combine with selection** - Best results from engineered + selected
- **Domain knowledge** - Survey question groups are meaningful

### Bayesian Optimization
- **Run after feature work** - Optimize on final feature set
- **Use all cores** - Set `n_jobs=-1` for faster search
- **More iterations = better** - 50-100 iterations ideal
- **Save everything** - Results guide future experiments

---

## 🔍 Troubleshooting

### Bayesian optimization runs out of memory
- Reduce `n_iter` from 30 to 20
- Reduce `cv_folds` from 5 to 3
- Disable feature engineering temporarily

### Feature selection takes too long
- Reduce RFE CV folds from 5 to 3
- Use smaller Random Forest (50 trees instead of 100)

### Feature engineering creates too many features
- Disable some feature types in config
- Use selected features: `use_selected_features: true`
- Reduce feature count with selection first

### Performance doesn't improve
- Check if SMOTE is enabled: `sampling.method: "smote"`
- Verify class weights in training
- Try different sampling methods with `compare_sampling.py`
- Ensure early stopping patience is not too low

---

## 📁 File Reference

### New Files Created
```
bayesian_optimization.py          - Hyperparameter optimization
feature_selection.py              - Feature importance analysis
src/data/feature_engineering.py   - Feature engineering module
ADVANCED_FEATURES_GUIDE.md        - This guide
```

### Modified Files
```
src/data/preprocessor.py          - Integrated feature engineering
config/config.yaml                - Added feature engineering config
requirements.txt                  - Added new dependencies
```

### Output Files
```
results/logs/bayesian_optimization.json     - Best hyperparameters
results/logs/bayesian_optimization_full.csv - All configurations
results/logs/feature_selection.json         - Selected features
results/logs/feature_scores.csv             - Feature importance scores
results/logs/rfe_cv_results.csv             - RFE optimization curve
results/plots/bayesian_optimization.png     - Optimization visualization
results/plots/feature_selection.png         - Feature analysis plots
```

---

## 🎓 Further Reading

- **Bayesian Optimization**: More efficient than grid search by modeling the objective function
- **Feature Selection**: Removes noise and reduces overfitting
- **Feature Engineering**: Domain knowledge + creativity = better features
- **Ensemble of Methods**: Combining multiple approaches yields best results

---

## ❓ Questions?

Check the code comments for detailed explanations of each method.

Good luck improving your model! 🚀

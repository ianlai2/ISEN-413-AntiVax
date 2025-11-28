# Anti-Vax Deep Learning Prediction Model

A comprehensive deep learning pipeline for predicting anti-vax sentiment (response Y) with cross-validation, dropout regularization, and reproducible results.

## Features

- **Deep Neural Network**: Multi-layer architecture with configurable hidden layers
- **Dropout Regularization**: Prevents overfitting with dropout layers after each hidden layer
- **K-Fold Cross-Validation**: 5-fold cross-validation for robust model evaluation
- **Reproducible Results**: Comprehensive seed setting across all random number generators
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **Model Checkpointing**: Saves best model during training
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, and AUC

## Project Structure

```
antivax-deep-learning/
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Data loading utilities
│   │   └── preprocessor.py      # Data preprocessing and scaling
│   ├── models/
│   │   ├── neural_network.py    # Neural network architecture with dropout
│   │   └── model_builder.py     # Model building utilities
│   ├── training/
│   │   ├── trainer.py           # Training pipeline with callbacks
│   │   └── cross_validator.py   # K-fold cross-validation
│   ├── evaluation/
│   │   └── metrics.py           # Evaluation metrics
│   └── utils/
│       ├── config.py            # Configuration management
│       └── seed.py              # Reproducibility utilities
├── config/
│   └── config.yaml              # Model and training configuration
├── data/
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed data files
├── models/
│   └── saved_models/            # Trained model checkpoints
├── results/
│   ├── logs/                    # Training logs and CV results
│   └── plots/                   # Visualization outputs
├── notebooks/
│   └── experiments.ipynb        # Exploratory notebooks
├── main.py                      # Main training script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
cd antivax-deep-learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your data files are in place:
   - Place `fully_imputed_training_data.csv` in the parent directory
   - Place `feature_metadata.csv` in the parent directory

## Configuration

Edit `config/config.yaml` to customize the model and training parameters:

```yaml
seed: 42  # Random seed for reproducibility

model:
  hidden_layers: [128, 64, 32]  # Hidden layer sizes
  dropout_rate: 0.3              # Dropout rate (0.0-1.0)
  activation: "relu"             # Activation function
  
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15
  
cross_validation:
  n_folds: 5
  shuffle: true
```

**📖 For detailed parameter tuning guidance**, see [PARAMETER_TUNING.md](PARAMETER_TUNING.md) - a comprehensive guide covering:
- All hyperparameters with ranges and effects
- Step-by-step tuning workflow
- Common issues and solutions
- Empirical results and recommendations

## Usage

### Training the Model

Run the main training script:

```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Perform 5-fold cross-validation
3. Train a final model on the full training set
4. Evaluate on the test set
5. Save the trained model and results

### Output Files

- **Model**: `models/saved_models/best_model.keras`
- **CV Results**: `results/logs/cv_results.json`
- **Training Logs**: Console output with detailed metrics

## Model Architecture

The neural network consists of:

1. **Input Layer**: Accepts all features from the dataset
2. **Hidden Layer 1**: 128 neurons, ReLU activation, Dropout (0.3)
3. **Hidden Layer 2**: 64 neurons, ReLU activation, Dropout (0.3)
4. **Hidden Layer 3**: 32 neurons, ReLU activation, Dropout (0.3)
5. **Output Layer**: 1 neuron, Sigmoid activation (binary classification)

### Regularization

- **Dropout**: Applied after each hidden layer (default: 30%)
- **Early Stopping**: Monitors validation loss with patience of 15 epochs
- **Learning Rate Reduction**: Reduces LR on plateau

## Reproducibility

All random operations are seeded for reproducibility:

- Python's `random` module
- NumPy's random number generator
- TensorFlow's random operations
- Environment variables for deterministic operations

Set the seed in `config/config.yaml` to ensure consistent results across runs.

## Evaluation Metrics

The model is evaluated using:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve
- **Misclassification Error**: Proportion of incorrectly classified samples (1 - Accuracy)

### Visualization

The pipeline generates 5 diagnostic plots in `results/plots/`:

1. **Confusion Matrix**: Visual breakdown of true/false positives/negatives
2. **ROC Curve**: Receiver Operating Characteristic with AUC score
3. **Training History**: Loss and accuracy curves over epochs
4. **Cross-Validation Results**: Performance distribution across folds
5. **Metrics Comparison**: CV performance vs final test performance

## Cross-Validation

The pipeline uses stratified K-fold cross-validation:

- **K-folds**: 5 (configurable)
- **Stratification**: Maintains class distribution in each fold
- **Shuffle**: Data is shuffled before splitting
- **Independent Models**: Each fold trains a fresh model instance

## Parameter Tuning Guide

This section provides detailed guidance on tuning hyperparameters for optimal performance.

### Architecture Parameters

#### **Hidden Layers** (`model.hidden_layers`)
- **Current Setting**: `[128, 64, 32]` (3 layers)
- **Range**: 1-5 layers with 16-512 neurons each
- **Effect**: 
  - More layers/neurons → Greater capacity, risk of overfitting
  - Fewer layers/neurons → Simpler model, risk of underfitting
- **Tuning Strategy**:
  - Start with `[64, 32]` for simple patterns
  - Use `[128, 64, 32]` for moderate complexity (current optimal)
  - Try `[256, 128, 64, 32]` for highly non-linear relationships
  - Ensure descending pattern (e.g., 128→64→32)
- **Empirical Results**: `[128, 64, 32]` achieved 86.90% CV accuracy

#### **Dropout Rate** (`model.dropout_rate`)
- **Current Setting**: `0.3` (30%)
- **Range**: 0.0 - 0.7
- **Effect**:
  - Higher dropout → Stronger regularization, reduced overfitting
  - Lower dropout → More capacity retained, risk of overfitting
- **Tuning Strategy**:
  - Start at 0.3 for balanced regularization
  - Increase to 0.4-0.5 if validation loss plateaus while training loss decreases
  - Decrease to 0.2 if both training and validation loss are high
  - Never exceed 0.7 (too much information loss)
- **Warning**: Dropout is only active during training, not inference

#### **Activation Function** (`model.activation`)
- **Current Setting**: `"relu"`
- **Options**: `"relu"`, `"elu"`, `"selu"`, `"tanh"`, `"leaky_relu"`
- **Effect**:
  - ReLU: Fast, simple, can suffer from dying neurons
  - ELU/SELU: Smooth, self-normalizing properties
  - Tanh: Outputs [-1, 1], can help with gradient flow
- **Recommendation**: ReLU is the standard choice; change only if experiencing gradient issues

### Training Parameters

#### **Learning Rate** (`training.learning_rate`)
- **Current Setting**: `0.001`
- **Range**: 1e-5 to 1e-2
- **Effect**:
  - Higher LR → Faster learning, risk of instability/overshooting
  - Lower LR → Slower learning, more stable convergence
- **Tuning Strategy**:
  - Use 0.001 as default (Adam optimizer default)
  - Try 0.0005 if training is unstable
  - Try 0.002-0.005 for faster initial convergence
  - Monitor for oscillating loss (sign LR is too high)
- **Adaptive Feature**: ReduceLROnPlateau automatically reduces LR by 50% when validation loss plateaus

#### **Batch Size** (`training.batch_size`)
- **Current Setting**: `32`
- **Range**: 16 - 256
- **Effect**:
  - Larger batches → Smoother gradients, faster training, more memory
  - Smaller batches → Noisier gradients, better generalization, less memory
- **Tuning Strategy**:
  - Use 32 for balanced performance
  - Reduce to 16 if memory constrained or dataset is small
  - Increase to 64-128 for larger datasets (>10,000 samples)
- **Trade-off**: Batch size affects gradient noise; some noise can help escape local minima

#### **Epochs** (`training.epochs`)
- **Current Setting**: `100`
- **Range**: 50 - 500
- **Effect**: Maximum training duration before stopping
- **Tuning Strategy**:
  - Set high (100-200) and rely on early stopping
  - Early stopping with patience=15 automatically stops when no improvement
  - If training consistently uses all epochs, increase the limit
- **Empirical Results**: Training typically stops around epoch 40-60 with current settings

#### **Early Stopping Patience** (`training.early_stopping_patience`)
- **Current Setting**: `15`
- **Range**: 5 - 30
- **Effect**: Number of epochs to wait for improvement before stopping
- **Tuning Strategy**:
  - Use 10-15 for most cases
  - Increase to 20-30 for noisy validation curves
  - Decrease to 5-10 for fast experimentation
- **Benefit**: Prevents overfitting and saves training time

#### **Validation Split** (`training.validation_split`)
- **Current Setting**: `0.2` (20%)
- **Range**: 0.1 - 0.3
- **Effect**: Proportion of training data used for validation
- **Tuning Strategy**:
  - Use 0.2 as standard
  - Increase to 0.3 for small datasets to better estimate performance
  - Decrease to 0.1 for very large datasets
- **Note**: Not used during cross-validation (explicit folds provided)

### Regularization Parameters

#### **Dropout** (primary regularization)
- Applied after each hidden layer
- See "Dropout Rate" section above

#### **Learning Rate Reduction**
- **Factor**: 0.5 (reduces LR by 50%)
- **Patience**: 5 epochs
- **Min LR**: 1e-7
- **Effect**: Helps fine-tune when approaching convergence

### Cross-Validation Parameters

#### **Number of Folds** (`cross_validation.n_folds`)
- **Current Setting**: `5`
- **Range**: 3 - 10
- **Effect**:
  - More folds → Better variance estimation, longer training
  - Fewer folds → Faster training, less reliable estimates
- **Tuning Strategy**:
  - Use 5 for balanced performance/reliability
  - Use 10 for small datasets or critical applications
  - Use 3 for quick experimentation
- **Standard**: 5-fold and 10-fold are most common in literature

### Data Preprocessing Parameters

#### **Feature Scaling**
- **Method**: StandardScaler (z-score normalization)
- **Effect**: Centers features to mean=0, std=1
- **Required**: Yes (neural networks are sensitive to feature scales)

#### **Feature Removal**
- **Constant Features**: Automatically removes 13 zero-variance features
- **Features Removed**: Q312_1, Q312_2, Q54_3, Q64_1-3, Q65_1, Q66_3, Q67_2-3, Q69_1-3
- **Effect**: Reduces dimensionality from 136 to 123 features
- **Benefit**: Eliminates uninformative features, reduces model complexity

### Optimizer Configuration

#### **Adam Optimizer** (current)
- **Beta1**: 0.9 (momentum)
- **Beta2**: 0.999 (RMSprop)
- **Epsilon**: 1e-7
- **Advantages**: Adaptive learning rates, momentum, generally robust

#### **Alternative Optimizers** (for experimentation)
- **SGD**: More stable but slower convergence
- **RMSprop**: Good for recurrent networks
- **AdamW**: Adam with weight decay (better generalization)

### Performance Tuning Workflow

1. **Baseline Establishment**
   - Train with default parameters
   - Record CV accuracy, AUC, and training time
   - Current baseline: 86.90% ± 0.81% CV accuracy

2. **Architecture Tuning** (most impactful)
   - Adjust hidden layers: try `[64, 32]` vs `[256, 128, 64]`
   - Monitor overfitting: gap between training and validation accuracy
   - Look for: CV accuracy > 85%, test accuracy within 2% of CV mean

3. **Regularization Tuning**
   - If overfitting (train >> val): increase dropout to 0.4-0.5
   - If underfitting (both low): decrease dropout to 0.2
   - Adjust early stopping patience if training is unstable

4. **Learning Rate Optimization**
   - If loss oscillates: reduce LR to 0.0005
   - If convergence is too slow: increase LR to 0.002
   - Monitor: loss should decrease smoothly

5. **Fine-Tuning**
   - Adjust batch size for memory/speed trade-offs
   - Increase epochs if early stopping triggers at maximum
   - Experiment with validation split for optimal monitoring

### Recommended Parameter Sets

#### **Fast Experimentation**
```yaml
model:
  hidden_layers: [64, 32]
  dropout_rate: 0.3
training:
  batch_size: 64
  epochs: 50
  early_stopping_patience: 10
cross_validation:
  n_folds: 3
```

#### **Balanced Performance** (current)
```yaml
model:
  hidden_layers: [128, 64, 32]
  dropout_rate: 0.3
training:
  batch_size: 32
  epochs: 100
  early_stopping_patience: 15
cross_validation:
  n_folds: 5
```

#### **Maximum Accuracy**
```yaml
model:
  hidden_layers: [256, 128, 64, 32]
  dropout_rate: 0.4
training:
  batch_size: 16
  epochs: 200
  learning_rate: 0.0005
  early_stopping_patience: 20
cross_validation:
  n_folds: 10
```

### Monitoring and Diagnostics

**Signs of Overfitting:**
- Training accuracy >> Validation accuracy (gap > 5%)
- Validation loss increases while training loss decreases
- High CV variance (std > 3%)
- **Solution**: Increase dropout, reduce model complexity, add more data

**Signs of Underfitting:**
- Both training and validation accuracy are low (< 80%)
- Loss plateaus early and remains high
- **Solution**: Increase model capacity, reduce dropout, train longer

**Optimal Training:**
- Training and validation curves converge
- CV std < 2%
- Test accuracy within 2% of CV mean
- Early stopping triggers before max epochs

## Example Output

```
============================================================
Cross-Validation Summary
============================================================
accuracy:
  Mean: 0.8542
  Std:  0.0123
auc:
  Mean: 0.9104
  Std:  0.0087
============================================================
Final Model Evaluation on Test Set
============================================================
Accuracy:  0.8623
Precision: 0.8512
Recall:    0.8734
F1 Score:  0.8621
AUC:       0.9187
```

## Troubleshooting

### Out of Memory Errors

- Reduce `batch_size` in config.yaml
- Reduce the number of neurons in `hidden_layers`

### Overfitting

- Increase `dropout_rate` (try 0.4 or 0.5)
- Reduce model complexity (fewer layers or neurons)
- Increase training data if possible

### Underfitting

- Decrease `dropout_rate`
- Increase model complexity (more layers or neurons)
- Increase `epochs`
- Decrease `learning_rate`

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please contact the project maintainers.
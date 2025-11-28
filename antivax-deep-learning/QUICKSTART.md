# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Model

### Option 1: Using Python directly
```bash
python main.py
```

### Option 2: Using the installed package (after running `pip install -e .`)
```bash
train-antivax
```

## Expected Output

The training process will:
1. Load and preprocess data
2. Run 5-fold cross-validation
3. Train final model on full training set
4. Evaluate on test set
5. Save model and results

## Output Files

- **Trained Model**: `models/saved_models/best_model.keras`
- **CV Results**: `results/logs/cv_results.json`

## Configuration

Edit `config/config.yaml` to adjust:
- Model architecture (hidden layers, dropout rate)
- Training parameters (batch size, epochs, learning rate)
- Cross-validation settings (number of folds)

## Key Features

✓ **Reproducible**: All random operations are seeded
✓ **Cross-Validation**: 5-fold stratified CV for robust evaluation
✓ **Dropout Regularization**: 30% dropout after each hidden layer
✓ **Early Stopping**: Prevents overfitting
✓ **Model Checkpointing**: Saves best model automatically

## Troubleshooting

### Import Errors
Make sure dependencies are installed:
```bash
pip install -r requirements.txt
```

### Data Not Found
Ensure `fully_imputed_training_data.csv` is in the parent directory or update the path in `config/config.yaml`

### Memory Issues
Reduce batch_size in config.yaml (try 16 or 8)

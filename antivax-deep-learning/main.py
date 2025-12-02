"""
Main script for training the anti-vax prediction deep learning model.
Includes cross-validation, dropout regularization, and reproducible results.
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.sampling import get_sampler
from src.models.neural_network import AntiVaxNN
from src.training.cross_validator import CrossValidator
from src.training.trainer import train_model_wrapper, Trainer
from src.evaluation.metrics import evaluate_model, detailed_evaluation, generate_all_plots
from feature_selection import comprehensive_feature_selection, save_feature_selection_results
from bayesian_optimization import run_bayesian_optimization


def main():
    """Main training pipeline."""
    
    # Load configuration
    print("Loading configuration...")
    config = load_config('config/config.yaml')

    # If bayesian optimization results exist, override config with best params
    bo_path = Path('results/logs/bayesian_optimization.json')
    best_params = None
    if bo_path.exists():
        try:
            import json
            with open(bo_path, 'r') as f:
                bo_data = json.load(f)
                best_params = bo_data.get('best_params', {})
            if best_params:
                print("\nApplying best hyperparameters from Bayesian Optimization...")
                # Map model hidden layers (including optional 4th and 5th layers)
                hl1 = best_params.get('model__hidden_layer_1')
                hl2 = best_params.get('model__hidden_layer_2')
                hl3 = best_params.get('model__hidden_layer_3')
                hl4 = best_params.get('model__hidden_layer_4', 0)
                hl5 = best_params.get('model__hidden_layer_5', 0)
                
                hidden_layers = []
                for hl in [hl1, hl2, hl3]:
                    if hl is not None:
                        hidden_layers.append(hl)
                # Add optional layers only if they are non-zero
                if hl4 is not None and hl4 > 0:
                    hidden_layers.append(hl4)
                if hl5 is not None and hl5 > 0:
                    hidden_layers.append(hl5)
                
                if hidden_layers:
                    config['model']['hidden_layers'] = hidden_layers
                # Dropout
                if 'model__dropout_rate' in best_params:
                    config['model']['dropout_rate'] = float(best_params['model__dropout_rate'])
                # Optimizer
                if 'model__optimizer' in best_params:
                    config['model']['optimizer'] = best_params['model__optimizer']
                # Learning rate
                if 'model__learning_rate' in best_params:
                    config['training']['learning_rate'] = float(best_params['model__learning_rate'])
                # Batch size / epochs
                if 'batch_size' in best_params:
                    config['training']['batch_size'] = int(best_params['batch_size'])
                if 'epochs' in best_params:
                    config['training']['epochs'] = int(best_params['epochs'])
        except Exception as e:
            print(f"Warning: Failed to load Bayesian optimization params: {e}")
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    data_path = config['data']['training_data']
    target_column = config['data']['target_column']
    
    loader = DataLoader(data_path, target_column)
    X, y = loader.split_features_target()
    
    # Preprocess data
    print("\n" + "="*60)
    print("Preprocessing Data")
    print("="*60)
    
    # Get sampling configuration
    sampling_config = config.get('sampling', {})
    sampling_method = sampling_config.get('method', 'none')
    
    # Create sampler if specified
    sampler = None
    if sampling_method != 'none':
        print(f"\nSampling method: {sampling_method}")
        sampler_kwargs = {
            'random_state': sampling_config.get('random_state', seed),
            'sampling_strategy': sampling_config.get('sampling_strategy', 'auto')
        }
        
        if sampling_method in ['smote', 'smote_tomek']:
            sampler_kwargs['k_neighbors'] = sampling_config.get('k_neighbors', 5)
        
        if sampling_method == 'smote_enn':
            sampler_kwargs['k_neighbors_smote'] = sampling_config.get('k_neighbors', 5)
            sampler_kwargs['k_neighbors_enn'] = sampling_config.get('k_neighbors_enn', 3)
        
        sampler = get_sampler(sampling_method, **sampler_kwargs)
    
    preprocessor = Preprocessor(scale_features=True, sampler=sampler)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        X, y, 
        test_size=0.2, 
        random_state=seed
    )

    # Optional feature selection on training data
    fs_cfg = config.get('feature_selection', {})
    if fs_cfg.get('enabled', False):
        print("\n" + "="*60)
        print("Feature Selection")
        print("="*60)
        # Build feature names list after preprocessing removals
        feature_names = [col for col in X.columns if col not in getattr(preprocessor, 'features_to_remove', [])]
        top_k = fs_cfg.get('top_k', 50)
        cv_folds_fs = fs_cfg.get('cv_folds', 5)
        results, f_scores, mi_scores, rf_scores, rfecv = comprehensive_feature_selection(
            X_train, y_train, feature_names, top_k=top_k, cv_folds=cv_folds_fs
        )
        save_feature_selection_results(results, f_scores, mi_scores, rf_scores, rfecv)
        consensus_feats = results.get('consensus_3plus', [])
        if consensus_feats:
            print(f"Using {len(consensus_feats)} consensus features from selection")
            name_to_idx = {name: i for i, name in enumerate(feature_names)}
            selected_idx = [name_to_idx[f] for f in consensus_feats if f in name_to_idx]
            # Subset train and test matrices
            X_train = X_train[:, selected_idx]
            X_test = X_test[:, selected_idx]
        else:
            print("No consensus features found; proceeding with all features.")
    
    # Get input dimensions
    input_dim = X_train.shape[1]
    print(f"\nInput features: {input_dim}")
    
    # Model builder function for cross-validation
    def build_model():
        """Build a new model instance."""
        nn = AntiVaxNN(
            input_dim=input_dim,
            hidden_layers=config['model']['hidden_layers'],
            dropout_rate=config['model']['dropout_rate'],
            activation=config['model']['activation'],
            output_activation=config['model']['output_activation'],
            learning_rate=config['training']['learning_rate'],
            optimizer=config['model'].get('optimizer', 'adam')
        )
        return nn.build()
    
    # Train function wrapper
    def train_func(model, X_tr, y_tr, X_val, y_val):
        """Training wrapper for cross-validation."""
        return train_model_wrapper(model, X_tr, y_tr, X_val, y_val, config)
    
    # Evaluate function wrapper
    def eval_func(model, X_val, y_val):
        """Evaluation wrapper for cross-validation."""
        return evaluate_model(model, X_val, y_val)
    
    # Optional: Run Bayesian Optimization prior to training to update config
    bo_cfg = config.get('bayesian_optimization', {})
    if bo_cfg.get('enabled', False):
        print("\n" + "="*60)
        print("Bayesian Optimization")
        print("="*60)
        try:
            run_bayesian_optimization(
                n_iter=bo_cfg.get('n_iter', 30),
                cv_folds=bo_cfg.get('cv_folds', 5),
                n_jobs=bo_cfg.get('n_jobs', 1),
                random_state=seed
            )
            # After BO, best params loader at start already applied when rerun; here we rely on prior override if present
            print("Bayesian optimization completed; best params saved to logs.")
        except Exception as e:
            print(f"Warning: Bayesian optimization failed: {e}")

    # Perform cross-validation
    print("\n" + "="*60)
    print("Cross-Validation")
    print("="*60)
    
    cv_config = config.get('cross_validation', {})
    cross_validator = CrossValidator(
        n_folds=cv_config.get('n_folds', 5),
        shuffle=cv_config.get('shuffle', True),
        random_state=seed
    )
    
    cv_results = cross_validator.cross_validate(
        X_train, y_train,
        model_builder=build_model,
        train_func=train_func,
        evaluate_func=eval_func,
        verbose=1
    )
    
    # Save cross-validation results
    cv_results_path = Path(config['paths']['logs']) / 'cv_results.json'
    cross_validator.save_results(str(cv_results_path))
    
    # Train final model on full training set
    print("\n" + "="*60)
    print("Training Final Model on Full Training Set")
    print("="*60)
    
    final_model = build_model()
    
    trainer = Trainer(
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        validation_split=config['training']['validation_split'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        verbose=1
    )
    
    # Create model checkpoint path
    model_save_path = Path(config['paths']['models']) / 'best_model.keras'
    if not model_save_path.parent.exists():
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Train final model
    history = trainer.train(
        final_model,
        X_train,
        y_train,
        model_checkpoint_path=str(model_save_path)
    )
    
    # Evaluate final model on test set
    print("\n" + "="*60)
    print("Final Model Evaluation on Test Set")
    print("="*60)
    
    test_results = detailed_evaluation(
        final_model,
        X_test,
        y_test,
        print_report=True
    )
    
    # Generate all diagnostic plots
    plots_dir = config['paths']['plots']
    generate_all_plots(cv_results, test_results, history, plots_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {model_save_path}")
    print(f"CV results saved to: {cv_results_path}")
    print(f"\nFinal Test Set Performance:")
    for metric, value in test_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    return {
        'cv_results': cv_results,
        'test_results': test_results,
        'model': final_model,
        'history': history
    }


if __name__ == '__main__':
    results = main()

"""
Hyperparameter optimization module for the AutoML text classification pipeline.

Uses Optuna for automated hyperparameter search across different model architectures
with pruning and early stopping for efficient optimization.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .config import Config
from .data_loader import DataProcessor
from .trainer import Trainer
from .evaluator import ModelEvaluator
from .models.ffn import FFNTextClassifier
from .models.cnn import CNNTextClassifier
from .models.transformer import TransformerTextClassifier
from .utils import setup_logger, get_device, seed_everything


class HPOOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, config: Config, device: Optional[str] = None, output_dir: Optional[Path] = None):
        """
        Initialize HPO optimizer.
        
        Args:
            config: Configuration object
            device: Device to use for training
            output_dir: Directory to save results (uses config default if None)
        """
        self.config = config
        # Ensure device is always a string for consistency with Trainer/Evaluator
        if device:
            self.device = device
        else:
            device_obj = get_device()
            self.device = str(device_obj)
        self.logger = setup_logger(__name__)
        
        # HPO configuration
        self.hpo_config = config.hpo
        self.n_trials = self.hpo_config['num_trials']
        self.timeout = self.hpo_config.get('timeout')
        self.metric = self.hpo_config['metric']
        self.direction = self.hpo_config['direction']
        
        # Data preparation
        self.data_processor = DataProcessor(config)
        self.data_cache = {}
        
        # Results storage
        self.best_trial = None
        self.best_model = None
        self.best_results = None
        self.optimization_history = []
        
        # Set output directory (use provided or default from config)
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(config.output['model_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_study(self) -> optuna.Study:
        """Setup Optuna study with appropriate sampler and pruner."""
        # Setup sampler
        sampler = TPESampler(seed=self.config.reproducibility['seed'])
        
        # Setup pruner
        if self.hpo_config.get('pruning', True):
            pruner_type = self.hpo_config.get('pruner', 'median')
            pruner_params = self.hpo_config.get('pruner_params', {})
            
            if pruner_type == 'median':
                pruner = MedianPruner(
                    n_startup_trials=pruner_params.get('n_startup_trials', 3),
                    n_warmup_steps=pruner_params.get('n_warmup_steps', 2),
                    interval_steps=pruner_params.get('interval_steps', 1)
                )
            else:
                # Use no pruning for unsupported types
                pruner = optuna.pruners.NopPruner()
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner
        )
        
        return study
    
    def _suggest_model_type(self, trial: optuna.Trial) -> str:
        """Suggest model architecture type."""
        # Available model types based on hyperparameter configuration
        available_models = list(self.config.models['hyperparameters'].keys())
        
        # TODO: Filter to only implemented models (for now)
        model_types = [model for model in available_models if model in ['ffn', 'cnn', 'transformer']]
        
        # Ensure we have at least one model type
        if not model_types:
            model_types = ['ffn', 'cnn']  # Fallback to basic models
        
        #TODO: Add BERT when implemented
        # if 'bert' in available_models:
        #     model_types.append('bert')
        
        return trial.suggest_categorical('model_type', model_types)
    
    def _suggest_hyperparameters(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters for the given model type."""
        model_config = self.config.models['hyperparameters'][model_type]
        hyperparams = {}
        
        # Special handling for transformer parameters to ensure embedding_dim % num_heads == 0
        if model_type == 'transformer':
            # First suggest num_heads
            num_heads_values = model_config.get('num_heads', [4, 8])
            num_heads = trial.suggest_categorical(f"{model_type}_num_heads", num_heads_values)
            hyperparams['num_heads'] = num_heads
            
            # Then suggest embedding_dim that's divisible by num_heads
            embedding_dim_range = model_config.get('embedding_dim', [128, 256])
            min_embed = min(embedding_dim_range)
            max_embed = max(embedding_dim_range)
            
            # Find valid embedding dimensions (divisible by num_heads)
            valid_dims = []
            for dim in range(min_embed, max_embed + 1):
                if dim % num_heads == 0:
                    valid_dims.append(dim)
            
            # If no valid dimensions found, use multiples of num_heads
            if not valid_dims:
                start_multiple = max(1, min_embed // num_heads)
                end_multiple = max_embed // num_heads + 1
                valid_dims = [num_heads * i for i in range(start_multiple, end_multiple + 1)]
            
            # Filter valid dims to only include those in the original range
            valid_dims = [dim for dim in valid_dims if dim in embedding_dim_range]
            if not valid_dims:
                # Fallback: use the original embedding_dim_range values that are divisible by num_heads
                valid_dims = [dim for dim in embedding_dim_range if dim % num_heads == 0]
                if not valid_dims:
                    # Last resort: use the closest multiples
                    valid_dims = [num_heads * (dim // num_heads) for dim in embedding_dim_range if dim >= num_heads]
                    valid_dims = list(set(valid_dims))  # Remove duplicates
            
            embedding_dim = trial.suggest_categorical(f"{model_type}_embedding_dim", valid_dims)
            hyperparams['embedding_dim'] = embedding_dim
            
            # Handle warmup parameters for transformers
            if 'warmup_steps' in model_config:
                warmup_steps_values = model_config['warmup_steps']
                if isinstance(warmup_steps_values, list):
                    warmup_steps = trial.suggest_categorical(f"{model_type}_warmup_steps", warmup_steps_values)
                    hyperparams['warmup_steps'] = warmup_steps
                    
            if 'warmup_ratio' in model_config:
                warmup_ratio_values = model_config['warmup_ratio']
                if isinstance(warmup_ratio_values, list):
                    if all(isinstance(v, float) for v in warmup_ratio_values):
                        warmup_ratio = trial.suggest_float(
                            f"{model_type}_warmup_ratio",
                            min(warmup_ratio_values),
                            max(warmup_ratio_values)
                        )
                    else:
                        warmup_ratio = trial.suggest_categorical(f"{model_type}_warmup_ratio", warmup_ratio_values)
                    hyperparams['warmup_ratio'] = warmup_ratio
            
            # Process remaining parameters normally
            for param_name, param_values in model_config.items():
                if param_name in ['num_heads', 'embedding_dim', 'warmup_steps', 'warmup_ratio']:
                    continue  # Already handled above
                    
                if isinstance(param_values, list):
                    if all(isinstance(v, int) for v in param_values):
                        # Integer parameters - use range suggestion
                        hyperparams[param_name] = trial.suggest_int(
                            f"{model_type}_{param_name}",
                            min(param_values),
                            max(param_values)
                        )
                    elif all(isinstance(v, float) for v in param_values):
                        # Float parameters - use range suggestion
                        hyperparams[param_name] = trial.suggest_float(
                            f"{model_type}_{param_name}",
                            min(param_values),
                            max(param_values)
                        )
                    else:
                        # Categorical parameters - convert nested lists to strings for Optuna
                        choices = []
                        for v in param_values:
                            if isinstance(v, list):
                                # Convert nested lists to comma-separated strings
                                choices.append(','.join(map(str, v)))
                            else:
                                choices.append(str(v))
                        
                        hyperparams[param_name] = trial.suggest_categorical(
                            f"{model_type}_{param_name}",
                            choices
                        )
                else:
                    # Single value parameters
                    hyperparams[param_name] = param_values
        else:
            # Standard parameter suggestion for non-transformer models
            for param_name, param_values in model_config.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, int) for v in param_values):
                        # Integer parameters - use range suggestion
                        hyperparams[param_name] = trial.suggest_int(
                            f"{model_type}_{param_name}",
                            min(param_values),
                            max(param_values)
                        )
                    elif all(isinstance(v, float) for v in param_values):
                        # Float parameters - use range suggestion
                        hyperparams[param_name] = trial.suggest_float(
                            f"{model_type}_{param_name}",
                            min(param_values),
                            max(param_values)
                        )
                    else:
                        # Categorical parameters - convert nested lists to strings for Optuna
                        choices = []
                        for v in param_values:
                            if isinstance(v, list):
                                # Convert nested lists to comma-separated strings
                                choices.append(','.join(map(str, v)))
                            else:
                                choices.append(str(v))
                        
                        hyperparams[param_name] = trial.suggest_categorical(
                            f"{model_type}_{param_name}",
                            choices
                        )
                else:
                    # Single value parameters
                    hyperparams[param_name] = param_values

        return hyperparams
    
    def _convert_hyperparams_for_model(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Convert hyperparameters from Optuna format to model format."""
        converted = {}
        for key, value in hyperparams.items():
            if isinstance(value, str) and ',' in value:
                # Convert comma-separated string back to list of integers
                try:
                    converted[key] = [int(x.strip()) for x in value.split(',')]
                except ValueError:
                    # If conversion fails, treat as regular string
                    converted[key] = value
            elif key == 'learning_rate' and isinstance(value, str):
                # Convert learning rate string to float (handles scientific notation)
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            elif isinstance(value, str) and ('e-' in value or 'E-' in value):
                # Convert scientific notation strings to float
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            elif isinstance(value, tuple):
                # Convert tuples back to lists for model compatibility
                converted[key] = list(value)
            else:
                converted[key] = value
        return converted
    
    def _create_model(self, model_type: str, hyperparams: Dict[str, Any], data_info: Dict[str, Any]) -> Any:
        """Create model instance based on type and hyperparameters."""
        # Ensure device is torch.device for model creation
        device = get_device()
        
        # Convert hyperparameters to model-compatible format
        model_hyperparams = self._convert_hyperparams_for_model(hyperparams)
        
        if model_type == 'ffn':
            return FFNTextClassifier(
                vocab_size=data_info['vocab_size'],
                num_classes=data_info['num_classes'],
                hyperparams=model_hyperparams,
                device=device
            )
        elif model_type == 'cnn':
            return CNNTextClassifier(
                vocab_size=data_info['vocab_size'],
                num_classes=data_info['num_classes'],
                hyperparams=model_hyperparams,
                device=device
            )
        elif model_type == 'transformer':
            return TransformerTextClassifier(
                vocab_size=data_info['vocab_size'],
                num_classes=data_info['num_classes'],
                hyperparams=model_hyperparams,
                device=device
            )
        # TODO: Add BERT models when implemented
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        try:
            # Set seed for reproducibility
            seed_everything(self.config.reproducibility['seed'])
            
            # Suggest model type and hyperparameters
            model_type = self._suggest_model_type(trial)
            hyperparams = self._suggest_hyperparameters(trial, model_type)
            
            self.logger.info(f"Trial {trial.number}: Testing {model_type} with {hyperparams}")
            
            # Prepare data for this model type (cache to avoid recomputation)
            if model_type not in self.data_cache:
                self.data_cache[model_type] = self.data_processor.prepare_data(model_type)
            
            data_info = self.data_cache[model_type]
            
            # Create model
            model = self._create_model(model_type, hyperparams, data_info)
            
            # Create trainer
            trainer = Trainer(
                config=self.config, 
                model=model, 
                model_type=model_type,
                hyperparams=hyperparams,
                device=self.device, 
                output_dir=self.output_dir
            )
            
            # Train model with pruning callback
            training_results = trainer.train(
                train_loader=data_info['train_loader'],
                val_loader=data_info['val_loader'],
                class_weights=data_info['class_weights']
            )
            
            # Report intermediate values for pruning
            for epoch, val_metrics in enumerate(training_results['training_history']['val']):
                # Map the configured metric to the actual trainer metric name
                intermediate_value = None
                
                # Create mapping from config metric names to trainer metric names
                metric_mapping = {
                    'f1_score_weighted': 'f1',
                    'f1_weighted': 'f1',
                    'precision_weighted': 'precision',
                    'recall_weighted': 'recall',
                    'f1_score_macro': 'f1',  # Note: trainer uses weighted, not macro
                    'accuracy': 'accuracy'
                }
                
                # Try to map the metric
                trainer_metric = metric_mapping.get(self.metric, self.metric)
                if trainer_metric in val_metrics:
                    intermediate_value = val_metrics[trainer_metric]
                else:
                    # Fallback to accuracy if metric not found
                    intermediate_value = val_metrics.get('accuracy', 0.0)
                
                trial.report(intermediate_value, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    self.logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
            
            # Get final metric value
            final_metrics = training_results['final_val_metrics']
            
            # Map the configured metric to the actual trainer metric name
            metric_mapping = {
                'f1_score_weighted': 'f1',
                'f1_weighted': 'f1',
                'precision_weighted': 'precision',
                'recall_weighted': 'recall',
                'f1_score_macro': 'f1',  # Note: trainer uses weighted, not macro
                'accuracy': 'accuracy'
            }
            
            # Try to map the metric
            trainer_metric = metric_mapping.get(self.metric, self.metric)
            if trainer_metric in final_metrics:
                objective_value = final_metrics[trainer_metric]
            else:
                # Fallback to accuracy if metric not found
                self.logger.warning(f"Metric {self.metric} (mapped to {trainer_metric}) not found in {list(final_metrics.keys())}. Using accuracy as fallback.")
                objective_value = final_metrics.get('accuracy', 0.0)
            
            # Store trial information
            trial_info = {
                'trial_number': trial.number,
                'model_type': model_type,
                'hyperparams': hyperparams,
                'objective_value': objective_value,
                'final_metrics': final_metrics,
                'training_time': training_results['total_time'],
                'total_epochs': training_results['total_epochs']
            }
            
            self.optimization_history.append(trial_info)
            
            self.logger.info(
                f"Trial {trial.number} completed: {model_type} - "
                f"{self.metric}={objective_value:.4f} in {training_results['total_epochs']} epochs"
            )
            
            return objective_value
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            # Return worst possible value to discourage this configuration
            return float('-inf') if self.direction == 'maximize' else float('inf')
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Optimization results with best trial information
        """
        self.logger.info(f"Starting HPO with {self.n_trials} trials")
        self.logger.info(f"Optimizing {self.metric} ({self.direction})")
        
        # Setup study
        study = self._setup_study()
        
        # Run optimization
        start_time = time.time()
        
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        optimization_time = time.time() - start_time
        
        # Get best trial
        self.best_trial = study.best_trial
        
        self.logger.info(f"HPO completed in {optimization_time:.1f}s")
        self.logger.info(f"Best trial: {self.best_trial.number}")
        self.logger.info(f"Best {self.metric}: {self.best_trial.value:.4f}")
        self.logger.info(f"Best params: {self.best_trial.params}")
        
        # Train final model with best hyperparameters
        self._train_best_model()
        
        # Evaluate on test set
        self._evaluate_best_model()
        
        # Prepare results
        results = {
            'best_trial': {
                'number': self.best_trial.number,
                'value': self.best_trial.value,
                'params': self.best_trial.params
            },
            'optimization_time': optimization_time,
            'total_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'study': study,
            'optimization_history': self.optimization_history
        }
        
        if self.best_results:
            results['test_metrics'] = self.best_results['metrics']
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _train_best_model(self) -> None:
        """Train the final model with best hyperparameters."""
        if self.best_trial is None:
            self.logger.error("No best trial available for final model training")
            return
            
        self.logger.info("Training final model with best hyperparameters")
        
        # Extract best configuration
        best_params = self.best_trial.params
        model_type = best_params['model_type']
        
        # Extract hyperparameters for the model
        hyperparams = {}
        prefix = f"{model_type}_"
        for key, value in best_params.items():
            if key.startswith(prefix):
                param_name = key[len(prefix):]
                hyperparams[param_name] = value
        
        # Convert hyperparameters to model-compatible format
        model_hyperparams = self._convert_hyperparams_for_model(hyperparams)
        
        # Prepare data
        if model_type not in self.data_cache:
            self.data_cache[model_type] = self.data_processor.prepare_data(model_type)
        
        data_info = self.data_cache[model_type]
        
        # Create and train final model
        self.best_model = self._create_model(model_type, hyperparams, data_info)
        trainer = Trainer(
            config=self.config, 
            model=self.best_model, 
            model_type=model_type,
            hyperparams=hyperparams,
            device=self.device, 
            output_dir=self.output_dir
        )
        
        training_results = trainer.train(
            train_loader=data_info['train_loader'],
            val_loader=data_info['val_loader'],
            class_weights=data_info['class_weights']
        )
        
        self.logger.info("Final model training completed")
    
    def _evaluate_best_model(self) -> None:
        """Evaluate the best model on test set."""
        if self.best_model is None:
            self.logger.warning("No best model available for evaluation")
            return
            
        if self.best_trial is None:
            self.logger.warning("No best trial available for evaluation")
            return
        
        self.logger.info("Evaluating best model on test set")
        
        # Get test data
        best_params = self.best_trial.params
        model_type = best_params['model_type']
        data_info = self.data_cache[model_type]
        
        # Evaluate model
        evaluator = ModelEvaluator(self.config, self.best_model, device=self.device)
        self.best_results = evaluator.evaluate(
            test_loader=data_info['test_loader'],
            label_names=None  # Could be extracted from label_encoder if needed
        )
        
        # Print summary
        evaluator.print_summary()
        
        # Save evaluation results
        evaluator.save_results(str(self.output_dir / 'best_model_evaluation.json'))
        
        self.logger.info("Model evaluation completed")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save optimization results."""
        import json
        
        # Prepare results for JSON serialization (exclude study object)
        serializable_results = {
            'best_trial': results['best_trial'],
            'optimization_time': results['optimization_time'],
            'total_trials': results['total_trials'],
            'completed_trials': results['completed_trials'],
            'pruned_trials': results['pruned_trials'],
            'failed_trials': results['failed_trials'],
            'optimization_history': results['optimization_history']
        }
        
        if 'test_metrics' in results:
            serializable_results['test_metrics'] = results['test_metrics']
        
        # Save to file
        results_path = self.output_dir / 'hpo_results.json'
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"HPO results saved to: {results_path}")
        
        # Save study object separately (for Optuna-specific analysis)
        study_path = self.output_dir / 'optuna_study.pkl'
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(results['study'], f)
        
        self.logger.info(f"Optuna study saved to: {study_path}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization process."""
        if not self.optimization_history:
            return {}
        
        # Analyze optimization history
        completed_trials = [t for t in self.optimization_history if 'objective_value' in t]
        
        if not completed_trials:
            return {}
        
        best_trial = max(completed_trials, key=lambda x: x['objective_value']) \
                    if self.direction == 'maximize' else min(completed_trials, key=lambda x: x['objective_value'])
        
        model_type_counts = {}
        for trial in completed_trials:
            model_type = trial['model_type']
            model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1
        
        return {
            'total_completed_trials': len(completed_trials),
            'best_trial_number': best_trial['trial_number'],
            'best_objective_value': best_trial['objective_value'],
            'best_model_type': best_trial['model_type'],
            'model_type_distribution': model_type_counts,
            'average_training_time': sum(t['training_time'] for t in completed_trials) / len(completed_trials),
            'optimization_metric': self.metric,
            'optimization_direction': self.direction
        }

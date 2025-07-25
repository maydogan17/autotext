"""
Main Pipeline orchestrator for the AutoML text classification system.

This module provides the high-level interface that coordinates all components:
- Configuration management
- Data loading and preprocessing  
- Model discovery and hyperparameter optimization
- Training and evaluation
- Results reporting and model persistence

The Pipeline class serves as the single entry point for the entire AutoML workflow.
"""

import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .config import Config
from .data_loader import DataProcessor
from .hpo import HPOOptimizer
from .trainer import Trainer
from .evaluator import ModelEvaluator
from .utils import setup_logger, get_device, seed_everything


class AutoMLPipeline:
    """
    Complete AutoML pipeline for text classification.
    
    This class orchestrates the entire machine learning workflow from data loading
    to model deployment, providing a simple interface for automated machine learning.
    """
    
    def __init__(self, config_path: Union[str, Path], device: Optional[str] = None):
        """
        Initialize the AutoML pipeline.
        
        Args:
            config_path: Path to the configuration YAML file
            device: Device to use for training (auto-detected if None)
        """
        # Load configuration
        self.config = Config(config_path)
        
        # Set up device
        self.device = device or str(get_device())
        
        # Set up logging
        self.logger = setup_logger(__name__)
        
        # Initialize components
        self.data_processor = None
        self.hpo_optimizer = None
        self.best_model = None
        self.best_trainer = None
        self.evaluator = None
        
        # Results storage
        self.results = {}
        self.pipeline_info = {
            'start_time': None,
            'end_time': None,
            'total_time': None,
            'status': 'initialized',
            'current_stage': None,
            'stages_completed': [],
            'config_path': str(config_path),
            'device': self.device
        }
        
        # Create timestamped output directory for this run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"run_{timestamp}"
        base_output_dir = Path(self.config.output['model_dir'])
        self.output_dir = base_output_dir / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"AutoML Pipeline initialized")
        self.logger.info(f"Configuration: {config_path}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def run(self, save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Run the complete AutoML pipeline.
        
        Args:
            save_artifacts: Whether to save models and results to disk
            
        Returns:
            Dictionary containing all pipeline results and metrics
        """
        self.pipeline_info['start_time'] = time.time()
        self.pipeline_info['status'] = 'running'
        
        try:
            self.logger.info("="*60)
            self.logger.info("STARTING AUTOML PIPELINE")
            self.logger.info("="*60)
            
            # Stage 1: Data Loading and Preprocessing
            self._run_stage("data_loading", self._load_and_preprocess_data)
            
            # Stage 2: Hyperparameter Optimization and Model Discovery
            self._run_stage("hyperparameter_optimization", self._run_hyperparameter_optimization)
            
            # Stage 3: Final Model Training
            self._run_stage("final_training", self._train_final_model)
            
            # Stage 4: Model Evaluation
            self._run_stage("evaluation", self._evaluate_model)
            
            # Stage 5: Save Results and Artifacts
            if save_artifacts:
                self._run_stage("saving_artifacts", self._save_artifacts)
            
            # Finalize pipeline
            self._finalize_pipeline()
            
            self.logger.info("="*60)
            self.logger.info("AUTOML PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)
            
            return self.results
            
        except Exception as e:
            self.pipeline_info['status'] = 'failed'
            self.pipeline_info['error'] = str(e)
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _run_stage(self, stage_name: str, stage_function):
        """Run a pipeline stage with logging and error handling."""
        self.pipeline_info['current_stage'] = stage_name
        self.logger.info(f"Starting stage: {stage_name}")
        
        stage_start = time.time()
        try:
            stage_function()
            stage_time = time.time() - stage_start
            
            self.pipeline_info['stages_completed'].append({
                'name': stage_name,
                'duration': stage_time,
                'status': 'completed'
            })
            
            self.logger.info(f"Stage '{stage_name}' completed in {stage_time:.1f}s")
            
        except Exception as e:
            stage_time = time.time() - stage_start
            self.pipeline_info['stages_completed'].append({
                'name': stage_name,
                'duration': stage_time,
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    def _load_and_preprocess_data(self):
        """Load and validate raw dataset without model-specific preprocessing."""
        self.logger.info(f"Loading dataset: {self.config.data['dataset_name']}")
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.config)
        
        # Load raw data to validate and get basic info
        train_df, test_df = self.data_processor._load_data()
        self.data_processor._validate_data(train_df, test_df)
        
        # Get basic dataset information without tokenization
        train_texts, val_texts, train_labels, val_labels = self.data_processor._split_data(train_df)
        
        # Calculate basic statistics
        total_train_samples = len(train_texts)
        total_val_samples = len(val_texts)
        total_test_samples = len(test_df)
        num_classes = len(self.data_processor.label_encoder.classes_)
        
        # Store basic data information (no model-specific details yet)
        self.results['data_info'] = {
            'dataset_name': self.config.data['dataset_name'],
            'num_classes': num_classes,
            'max_length': self.config.data['max_length'],
            'train_samples': total_train_samples,
            'val_samples': total_val_samples,
            'test_samples': total_test_samples,
            'total_samples': total_train_samples + total_val_samples + total_test_samples,
            'class_names': [str(cls) for cls in self.data_processor.label_encoder.classes_]
        }
        
        self.logger.info(f"Dataset validated successfully:")
        self.logger.info(f"  - Total samples: {self.results['data_info']['total_samples']}")
        self.logger.info(f"  - Train samples: {self.results['data_info']['train_samples']}")
        self.logger.info(f"  - Validation samples: {self.results['data_info']['val_samples']}")
        self.logger.info(f"  - Test samples: {self.results['data_info']['test_samples']}")
        self.logger.info(f"  - Number of classes: {self.results['data_info']['num_classes']}")
        self.logger.info(f"  - Max sequence length: {self.results['data_info']['max_length']}")
        self.logger.info("Note: Model-specific tokenization will be done during HPO for each model type")
    
    def _run_hyperparameter_optimization(self):
        """Run hyperparameter optimization to find the best model and hyperparameters."""
        self.logger.info("Starting hyperparameter optimization...")
        self.logger.info(f"Number of trials: {self.config.hpo['num_trials']}")
        self.logger.info(f"Optimization metric: {self.config.hpo['metric']}")
        
        # Set seed for reproducibility
        seed_everything(self.config.reproducibility['seed'])
        
        # Initialize HPO optimizer with our output directory
        self.hpo_optimizer = HPOOptimizer(self.config, device=self.device, output_dir=self.output_dir)
        
        # Run optimization
        hpo_results = self.hpo_optimizer.optimize()
        
        # Store HPO results
        self.results['hpo_results'] = {
            'best_trial': hpo_results['best_trial'],
            'optimization_time': hpo_results['optimization_time'],
            'total_trials': hpo_results['total_trials'],
            'completed_trials': hpo_results['completed_trials'],
            'pruned_trials': hpo_results['pruned_trials'],
            'failed_trials': hpo_results['failed_trials'],
            'optimization_summary': self.hpo_optimizer.get_optimization_summary()
        }
        
        # Get the best model from HPO
        self.best_model = self.hpo_optimizer.best_model
        
        self.logger.info("Hyperparameter optimization completed:")
        self.logger.info(f"  - Best model: {hpo_results['best_trial']['params']['model_type']}")
        self.logger.info(f"  - Best {self.config.hpo['metric']}: {hpo_results['best_trial']['value']:.4f}")
        self.logger.info(f"  - Optimization time: {hpo_results['optimization_time']:.1f}s")
    
    def _train_final_model(self):
        """Train the final model with best hyperparameters (already done in HPO)."""
        if self.best_model is None:
            raise ValueError("No best model available. HPO must be run first.")
        
        if self.hpo_optimizer is None:
            raise ValueError("HPO optimizer not available.")
            
        if not hasattr(self.hpo_optimizer, 'best_trial') or getattr(self.hpo_optimizer, 'best_trial', None) is None:
            raise ValueError("No best trial available from HPO.")
        
        self.logger.info("Final model training completed during HPO")
        
        # The model is already trained during HPO, so we just need to store the info
        best_trial = getattr(self.hpo_optimizer, 'best_trial', None)
        if best_trial is None:
            raise ValueError("Best trial is not available")
            
        best_params = best_trial.params
        model_type = best_params['model_type']
        
        self.results['final_model'] = {
            'model_type': model_type,
            'hyperparameters': {k.replace(f'{model_type}_', ''): v for k, v in best_params.items() 
                              if k.startswith(f'{model_type}_')},
            'training_completed': True
        }
    
    def _evaluate_model(self):
        """Evaluate the final model on the test set."""
        if self.best_model is None:
            raise ValueError("No best model available for evaluation")
        
        if self.hpo_optimizer is None:
            raise ValueError("HPO optimizer not available.")
            
        if not hasattr(self.hpo_optimizer, 'best_trial') or getattr(self.hpo_optimizer, 'best_trial', None) is None:
            raise ValueError("No best trial available from HPO.")
        
        self.logger.info("Evaluating final model on test set...")
        
        # Get evaluation results from HPO (already computed)
        best_results = getattr(self.hpo_optimizer, 'best_results', None)
        if best_results is not None and best_results:
            evaluation_results = best_results
        else:
            # Fallback: run evaluation if not available
            self.evaluator = ModelEvaluator(self.config, self.best_model, device=self.device)
            
            # Get test data using the same lazy loading approach as HPO
            best_trial = getattr(self.hpo_optimizer, 'best_trial', None)
            if best_trial is None:
                raise ValueError("Best trial is not available")
                
            best_params = best_trial.params
            model_type = best_params['model_type']
            
            # Check HPO data cache first
            data_cache = getattr(self.hpo_optimizer, 'data_cache', None)
            if data_cache is not None and model_type in data_cache:
                data_info = data_cache[model_type]
                self.logger.info(f"Using cached data for model type: {model_type}")
            else:
                # Recreate data with proper tokenizer for this model type
                if self.data_processor is None:
                    raise ValueError("Data processor not available")
                self.logger.info(f"Preparing fresh data for model type: {model_type}")
                data_info = self.data_processor.prepare_data(model_type)
            
            evaluation_results = self.evaluator.evaluate(
                test_loader=data_info['test_loader'],
                label_names=None
            )
        
        # Store evaluation results
        self.results['evaluation'] = {
            'test_metrics': evaluation_results['metrics'],
            'confusion_matrix': evaluation_results.get('confusion_matrix', {}).tolist() if 'confusion_matrix' in evaluation_results else None,
            'classification_report': evaluation_results.get('classification_report', {}),
            'per_class_metrics': evaluation_results.get('per_class_metrics', {})
        }
        
        # Log key metrics
        metrics = evaluation_results['metrics']
        self.logger.info("Final model evaluation:")
        self.logger.info(f"  - Accuracy: {metrics.get('accuracy', 0):.4f}")
        self.logger.info(f"  - F1 (weighted): {metrics.get('f1_weighted', 0):.4f}")
        self.logger.info(f"  - F1 (macro): {metrics.get('f1_macro', 0):.4f}")
        self.logger.info(f"  - Precision (weighted): {metrics.get('precision_weighted', 0):.4f}")
        self.logger.info(f"  - Recall (weighted): {metrics.get('recall_weighted', 0):.4f}")
    
    def _save_artifacts(self):
        """Save models, results, and other artifacts."""
        self.logger.info("Saving artifacts and results...")
        
        # Save configuration
        config_path = self.output_dir / self.config.output['config_filename']
        self.config.save(config_path)
        
        # Save complete results
        results_path = self.output_dir / 'pipeline_results.json'
        self._save_results_to_json(results_path)
        
        # Generate summary report
        summary_path = self.output_dir / 'pipeline_summary.txt'
        self._generate_summary_report(summary_path)
        
        self.logger.info(f"Artifacts saved to: {self.output_dir}")
        self.logger.info(f"  - Configuration: {config_path}")
        self.logger.info(f"  - Results: {results_path}")
        self.logger.info(f"  - Summary: {summary_path}")
    
    def _finalize_pipeline(self):
        """Finalize the pipeline execution."""
        self.pipeline_info['end_time'] = time.time()
        self.pipeline_info['total_time'] = self.pipeline_info['end_time'] - self.pipeline_info['start_time']
        self.pipeline_info['status'] = 'completed'
        self.pipeline_info['current_stage'] = None
        
        # Add pipeline info to results
        self.results['pipeline_info'] = self.pipeline_info
        
        self.logger.info(f"Pipeline completed in {self.pipeline_info['total_time']:.1f}s")
    
    def _save_results_to_json(self, output_path: Path):
        """Save complete results to JSON file."""
        # Create a JSON-serializable version of results
        json_results = self._make_json_serializable(self.results.copy())
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _generate_summary_report(self, output_path: Path):
        """Generate a human-readable summary report."""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("AUTOML PIPELINE SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Pipeline info
            f.write("PIPELINE INFORMATION\n")
            f.write("-"*40 + "\n")
            total_time = self.pipeline_info.get('total_time', 0)
            if total_time is None:
                total_time = 0
            f.write(f"Total execution time: {total_time:.1f} seconds\n")
            f.write(f"Device used: {self.device}\n")
            f.write(f"Configuration file: {self.pipeline_info['config_path']}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            # Data info
            if 'data_info' in self.results:
                data_info = self.results['data_info']
                f.write("DATASET INFORMATION\n")
                f.write("-"*40 + "\n")
                f.write(f"Dataset: {data_info['dataset_name']}\n")
                f.write(f"Total samples: {data_info['total_samples']}\n")
                f.write(f"Number of classes: {data_info['num_classes']}\n")
                f.write(f"Max sequence length: {data_info['max_length']}\n")
                if 'class_names' in data_info:
                    f.write(f"Class names: {', '.join(data_info['class_names'])}\n")
                f.write("\n")
            
            # HPO results
            if 'hpo_results' in self.results:
                hpo = self.results['hpo_results']
                f.write("HYPERPARAMETER OPTIMIZATION\n")
                f.write("-"*40 + "\n")
                f.write(f"Total trials: {hpo['total_trials']}\n")
                f.write(f"Completed trials: {hpo['completed_trials']}\n")
                f.write(f"Pruned trials: {hpo['pruned_trials']}\n")
                f.write(f"Failed trials: {hpo['failed_trials']}\n")
                f.write(f"Optimization time: {hpo['optimization_time']:.1f} seconds\n")
                f.write(f"Best model: {hpo['best_trial']['params']['model_type']}\n")
                f.write(f"Best score: {hpo['best_trial']['value']:.4f}\n\n")
            
            # Final model
            if 'final_model' in self.results:
                model = self.results['final_model']
                f.write("FINAL MODEL\n")
                f.write("-"*40 + "\n")
                f.write(f"Model type: {model['model_type']}\n")
                f.write("Hyperparameters:\n")
                for param, value in model['hyperparameters'].items():
                    f.write(f"  - {param}: {value}\n")
                f.write("\n")
            
            # Evaluation results
            if 'evaluation' in self.results:
                metrics = self.results['evaluation']['test_metrics']
                f.write("FINAL EVALUATION RESULTS\n")
                f.write("-"*40 + "\n")
                f.write(f"Accuracy: {metrics.get('accuracy', 0):.4f}\n")
                f.write(f"F1 Score (weighted): {metrics.get('f1_weighted', 0):.4f}\n")
                f.write(f"F1 Score (macro): {metrics.get('f1_macro', 0):.4f}\n")
                f.write(f"Precision (weighted): {metrics.get('precision_weighted', 0):.4f}\n")
                f.write(f"Recall (weighted): {metrics.get('recall_weighted', 0):.4f}\n")
                if 'roc_auc' in metrics:
                    f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("End of Report\n")
            f.write("="*80 + "\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of the pipeline results."""
        if not self.results:
            return {"status": "not_run", "message": "Pipeline has not been executed yet"}
        
        total_time = self.pipeline_info.get('total_time', 0)
        if total_time is None:
            total_time = 0
            
        summary = {
            "status": self.pipeline_info.get('status', 'unknown'),
            "execution_time": total_time,
            "device": self.device
        }
        
        if 'data_info' in self.results:
            summary['dataset'] = {
                'name': self.results['data_info']['dataset_name'],
                'samples': self.results['data_info']['total_samples'],
                'classes': self.results['data_info']['num_classes']
            }
        
        if 'hpo_results' in self.results:
            hpo = self.results['hpo_results']
            summary['optimization'] = {
                'best_model': hpo['best_trial']['params']['model_type'],
                'best_score': hpo['best_trial']['value'],
                'trials': hpo['total_trials'],
                'time': hpo['optimization_time']
            }
        
        if 'evaluation' in self.results:
            metrics = self.results['evaluation']['test_metrics']
            summary['performance'] = {
                'accuracy': metrics.get('accuracy', 0),
                'f1_weighted': metrics.get('f1_weighted', 0),
                'f1_macro': metrics.get('f1_macro', 0)
            }
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of the pipeline results."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("AUTOML PIPELINE SUMMARY")
        print("="*60)
        
        print(f"Status: {summary['status']}")
        print(f"Execution Time: {summary['execution_time']:.1f}s")
        print(f"Device: {summary['device']}")
        
        if 'dataset' in summary:
            print(f"\nDataset: {summary['dataset']['name']}")
            print(f"Samples: {summary['dataset']['samples']}")
            print(f"Classes: {summary['dataset']['classes']}")
        
        if 'optimization' in summary:
            print(f"\nBest Model: {summary['optimization']['best_model']}")
            print(f"Best Score: {summary['optimization']['best_score']:.4f}")
            print(f"Trials: {summary['optimization']['trials']}")
            print(f"HPO Time: {summary['optimization']['time']:.1f}s")
        
        if 'performance' in summary:
            print(f"\nFinal Performance:")
            print(f"  Accuracy: {summary['performance']['accuracy']:.4f}")
            print(f"  F1 (weighted): {summary['performance']['f1_weighted']:.4f}")
            print(f"  F1 (macro): {summary['performance']['f1_macro']:.4f}")
        
        print("="*60)

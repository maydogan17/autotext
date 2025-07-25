"""
Evaluation module for the AutoML text classification pipeline.

Handles model evaluation with comprehensive metrics, confusion matrices,
classification reports, and error analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)

from .config import Config
from .models.base import BaseTextClassifier
from .utils import setup_logger, get_device


class ModelEvaluator:
    """Comprehensive model evaluation for text classification."""
    
    def __init__(self, config: Config, model: BaseTextClassifier, device: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration object
            model: Trained model to evaluate
            device: Device to use for evaluation
        """
        self.config = config
        self.model = model
        self.device = device or get_device()
        self.logger = setup_logger(__name__)
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Evaluation results storage
        self.predictions = []
        self.probabilities = []
        self.true_labels = []
        self.evaluation_results = {}
    
    def evaluate(
        self, 
        test_loader: DataLoader,
        label_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            label_names: Optional list of class names for reports
            
        Returns:
            Comprehensive evaluation results
        """
        self.logger.info("Starting model evaluation")
        
        # Reset previous results
        self.predictions = []
        self.probabilities = []
        self.true_labels = []
        
        # Run inference
        self._run_inference(test_loader)
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Generate reports
        reports = self._generate_reports(label_names)
        
        # Combine results
        self.evaluation_results = {
            'metrics': metrics,
            'reports': reports,
            'predictions': self.predictions,
            'probabilities': self.probabilities,
            'true_labels': self.true_labels
        }
        
        self.logger.info(f"Evaluation completed - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}")
        
        return self.evaluation_results
    
    def _run_inference(self, test_loader: DataLoader) -> None:
        """Run model inference on test data."""
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Get probabilities and predictions
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                self.predictions.extend(predictions.cpu().numpy())
                self.probabilities.extend(probabilities.cpu().numpy())
                self.true_labels.extend(labels.cpu().numpy())
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        # Basic metrics
        accuracy = accuracy_score(self.true_labels, self.predictions)
        
        # Multi-class metrics with different averaging strategies
        f1_macro = f1_score(self.true_labels, self.predictions, average='macro', zero_division=0)
        f1_micro = f1_score(self.true_labels, self.predictions, average='micro', zero_division=0)
        f1_weighted = f1_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        
        precision_macro = precision_score(self.true_labels, self.predictions, average='macro', zero_division=0)
        precision_micro = precision_score(self.true_labels, self.predictions, average='micro', zero_division=0)
        precision_weighted = precision_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        
        recall_macro = recall_score(self.true_labels, self.predictions, average='macro', zero_division=0)
        recall_micro = recall_score(self.true_labels, self.predictions, average='micro', zero_division=0)
        recall_weighted = recall_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_micro': float(f1_micro), 
            'f1_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'precision_micro': float(precision_micro),
            'precision_weighted': float(precision_weighted),
            'recall_macro': float(recall_macro),
            'recall_micro': float(recall_micro),
            'recall_weighted': float(recall_weighted)
        }
        
        # Add ROC AUC for binary/multiclass if possible
        try:
            num_classes = len(set(self.true_labels))
            if num_classes == 2:
                # Binary classification - use probabilities for positive class
                roc_auc = roc_auc_score(self.true_labels, [prob[1] for prob in self.probabilities])
                metrics['roc_auc'] = float(roc_auc)
            elif num_classes > 2:
                # Multiclass - use one-vs-rest strategy
                roc_auc = roc_auc_score(
                    self.true_labels, 
                    self.probabilities, 
                    multi_class='ovr', 
                    average='weighted'
                )
                metrics['roc_auc'] = float(roc_auc)
        except Exception as e:
            self.logger.debug(f"Could not compute ROC AUC: {e}")
        
        return metrics
    
    def _generate_reports(self, label_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate detailed classification reports."""
        reports = {}
        
        # Classification report
        try:
            classification_rep = classification_report(
                self.true_labels,
                self.predictions,
                target_names=label_names,
                output_dict=True,
                zero_division=0
            )
            reports['classification_report'] = classification_rep
        except Exception as e:
            self.logger.warning(f"Could not generate classification report: {e}")
        
        # Confusion matrix
        try:
            cm = confusion_matrix(self.true_labels, self.predictions)
            reports['confusion_matrix'] = cm.tolist()
            
            # Normalized confusion matrix
            cm_normalized = confusion_matrix(
                self.true_labels, 
                self.predictions, 
                normalize='true'
            )
            reports['confusion_matrix_normalized'] = cm_normalized.tolist()
        except Exception as e:
            self.logger.warning(f"Could not generate confusion matrix: {e}")
        
        # Per-class metrics
        try:
            unique_labels = sorted(set(self.true_labels))
            per_class_metrics = {}
            
            for label in unique_labels:
                label_name = label_names[label] if label_names and label < len(label_names) else f"class_{label}"
                
                # Binary metrics for this class vs all others
                y_true_binary = [1 if y == label else 0 for y in self.true_labels]
                y_pred_binary = [1 if y == label else 0 for y in self.predictions]
                
                per_class_metrics[label_name] = {
                    'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
                    'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
                    'f1': float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
                    'support': int(sum(y_true_binary))
                }
            
            reports['per_class_metrics'] = per_class_metrics
        except Exception as e:
            self.logger.warning(f"Could not generate per-class metrics: {e}")
        
        return reports
    
    def get_prediction_errors(
        self, 
        texts: Optional[List[str]] = None,
        max_errors: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get examples of prediction errors for analysis.
        
        Args:
            texts: Optional list of original texts
            max_errors: Maximum number of errors to return
            
        Returns:
            List of error examples with details
        """
        if not self.predictions or not self.true_labels:
            self.logger.warning("No predictions available. Run evaluate() first.")
            return []
        
        errors = []
        
        for i, (true_label, pred_label) in enumerate(zip(self.true_labels, self.predictions)):
            if true_label != pred_label and len(errors) < max_errors:
                error_info = {
                    'index': i,
                    'true_label': int(true_label),
                    'predicted_label': int(pred_label),
                    'confidence': float(max(self.probabilities[i])),
                    'probabilities': [float(p) for p in self.probabilities[i]]
                }
                
                if texts and i < len(texts):
                    error_info['text'] = texts[i]
                
                errors.append(error_info)
        
        # Sort by confidence (most confident wrong predictions first)
        errors.sort(key=lambda x: x['confidence'], reverse=True)
        
        return errors
    
    def save_results(self, output_path: Optional[str] = None) -> None:
        """
        Save evaluation results to file.
        
        Args:
            output_path: Optional custom output path
        """
        if not self.evaluation_results:
            self.logger.warning("No evaluation results to save. Run evaluate() first.")
            return
        
        if output_path is None:
            output_dir = Path(self.config.output['model_dir'])
            output_file = output_dir / 'evaluation_results.json'
        else:
            output_file = Path(output_path)
        
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results_to_save = {
            'metrics': self.evaluation_results['metrics'],
            'reports': self.evaluation_results['reports']
        }
        
        # Save to JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to: {output_file}")
    
    def print_summary(self, label_names: Optional[List[str]] = None) -> None:
        """Print a summary of evaluation results."""
        if not self.evaluation_results:
            self.logger.warning("No evaluation results available. Run evaluate() first.")
            return
        
        metrics = self.evaluation_results['metrics']
        
        print("\n" + "="*50)
        print("MODEL EVALUATION SUMMARY")
        print("="*50)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
        print(f"  Precision:     {metrics['precision_weighted']:.4f}")
        print(f"  Recall:        {metrics['recall_weighted']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  ROC AUC:       {metrics['roc_auc']:.4f}")
        
        # Print confusion matrix if available
        if 'confusion_matrix' in self.evaluation_results['reports']:
            cm = np.array(self.evaluation_results['reports']['confusion_matrix'])
            print(f"\nConfusion Matrix:")
            print(cm)
        
        # Print per-class metrics if available
        if 'per_class_metrics' in self.evaluation_results['reports']:
            print(f"\nPer-Class Metrics:")
            per_class = self.evaluation_results['reports']['per_class_metrics']
            
            print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
            print("-" * 60)
            
            for class_name, class_metrics in per_class.items():
                print(f"{class_name:<15} {class_metrics['precision']:<10.3f} "
                      f"{class_metrics['recall']:<10.3f} {class_metrics['f1']:<10.3f} "
                      f"{class_metrics['support']:<10}")
        
        print("="*50)
    
    def compare_models(self, other_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare this model's results with another model's results.
        
        Args:
            other_results: Results from another model evaluation
            
        Returns:
            Comparison metrics and analysis
        """
        if not self.evaluation_results:
            self.logger.warning("No evaluation results available. Run evaluate() first.")
            return {}
        
        current_metrics = self.evaluation_results['metrics']
        other_metrics = other_results.get('metrics', {})
        
        comparison = {}
        
        for metric_name in current_metrics:
            if metric_name in other_metrics:
                current_value = current_metrics[metric_name]
                other_value = other_metrics[metric_name]
                difference = current_value - other_value
                
                comparison[metric_name] = {
                    'current': current_value,
                    'other': other_value,
                    'difference': difference,
                    'better': difference > 0
                }
        
        return comparison

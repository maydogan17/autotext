"""
Training module for the AutoML text classification pipeline.

Handles model training with early stopping, learning rate scheduling,
checkpointing, and metrics tracking for all model types.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .config import Config
from .models.base import BaseTextClassifier
from .schedulers import WarmupLRScheduler, AdaptiveLRScheduler, create_optimizer_and_scheduler
from .utils import setup_logger, get_device
from .schedulers import create_optimizer_and_scheduler


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """
        Update metrics with batch results.
        
        Args:
            predictions: Model predictions (logits)
            targets: True labels
            loss: Batch loss
        """
        pred_classes = torch.argmax(predictions, dim=1)
        self.predictions.extend(pred_classes.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0.0
        
        if not self.predictions or not self.targets:
            return {'loss': avg_loss, 'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        accuracy = accuracy_score(self.targets, self.predictions)
        f1 = f1_score(self.targets, self.predictions, average='weighted', zero_division=0)
        precision = precision_score(self.targets, self.predictions, average='weighted', zero_division=0)
        recall = recall_score(self.targets, self.predictions, average='weighted', zero_division=0)
        
        return {
            'loss': float(avg_loss),
            'accuracy': float(accuracy),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        }


class Trainer:
    """Main trainer class for text classification models."""
    
    def __init__(
        self, 
        config: Config, 
        model: BaseTextClassifier, 
        model_type: str,
        hyperparams: Dict[str, Any],
        device: Optional[str] = None, 
        output_dir: Optional[Path] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            model: Model to train
            model_type: Type of model (for scheduler optimization)
            hyperparams: Model hyperparameters (may include learning rate)
            device: Device to use for training
            output_dir: Directory to save models (uses config default if None)
        """
        self.config = config
        self.model = model
        self.model_type = model_type
        self.hyperparams = hyperparams
        self.device = device or get_device()
        self.logger = setup_logger(__name__)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training configuration
        self.training_config = config.training
        self.max_epochs = self.training_config['max_epochs']
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {'train': [], 'val': []}
        self.total_steps = None
        self.steps_per_epoch = None
        
        # Setup early stopping
        early_stopping_config = self.training_config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 7),
            min_delta=early_stopping_config.get('min_delta', 0.001),
            restore_best_weights=early_stopping_config.get('restore_best_weights', True)
        )
        
        # Model saving
        if output_dir is not None:
            self.model_dir = Path(output_dir)
        else:
            self.model_dir = Path(config.output['model_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer_and_scheduler(self, total_steps: int, steps_per_epoch: int):
        """Setup optimizer and scheduler with knowledge of training duration."""
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        
                # Initialize optimizer and scheduler - will be set by create_optimizer_and_scheduler
        self.optimizer: torch.optim.Optimizer
        self.scheduler: Optional[Union[ReduceLROnPlateau, StepLR, WarmupLRScheduler, AdaptiveLRScheduler]]
        
        # Use the new adaptive optimizer and scheduler creation
        optimizer, scheduler = create_optimizer_and_scheduler(
            model=self.model,
            model_type=self.model_type,
            hyperparams=self.hyperparams,
            config=self.training_config,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch
        )
        
        # Ensure optimizer was created successfully
        assert optimizer is not None, "Failed to create optimizer"
        self.optimizer = optimizer
        self.scheduler = scheduler
    

    
    def train_epoch(self, train_loader: DataLoader, class_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            class_weights: Class weights for loss calculation
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        metrics_tracker = MetricsTracker()
        
        # Setup loss function
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            metrics_tracker.update(outputs.detach(), labels, loss.item())
            
            # Log batch progress
            if batch_idx % 50 == 0:
                self.logger.debug(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        return metrics_tracker.compute()
    
    def validate_epoch(self, val_loader: DataLoader, class_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            class_weights: Class weights for loss calculation
            
        Returns:
            Validation metrics for the epoch
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        # Setup loss function
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Update metrics
                metrics_tracker.update(outputs, labels, loss.item())
        
        return metrics_tracker.compute()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            class_weights: Class weights for loss calculation
            
        Returns:
            Training results and metrics
        """
        self.logger.info(f"Starting training for {self.max_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        
        # Setup optimizer and scheduler now that we have data loader info
        total_steps = self.max_epochs * len(train_loader)
        steps_per_epoch = len(train_loader)
        self._setup_optimizer_and_scheduler(total_steps, steps_per_epoch)
        
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, class_weights)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader, class_weights)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Store metrics
            self.training_history['train'].append(train_metrics)
            self.training_history['val'].append(val_metrics)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch+1}/{self.max_epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt')
            
            # Check early stopping
            if self.early_stopping(val_metrics['loss'], self.model):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s")
        
        # Load best weights if early stopping was used
        if self.early_stopping.restore_best_weights:
            best_model_path = self.model_dir / 'best_model.pt'
            if best_model_path.exists():
                self.load_checkpoint('best_model.pt')
                self.logger.info("Loaded best model weights")
        
        return {
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_time': total_time,
            'final_train_metrics': self.training_history['train'][-1],
            'final_val_metrics': self.training_history['val'][-1]
        }
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.model_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'model_config': self.model.hyperparams
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.model_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training results.
        
        Returns:
            Training summary dictionary
        """
        if not self.training_history['train']:
            return {}
        
        best_epoch = min(range(len(self.training_history['val'])), 
                        key=lambda i: self.training_history['val'][i]['loss'])
        
        return {
            'total_epochs': len(self.training_history['train']),
            'best_epoch': best_epoch + 1,
            'best_val_loss': self.training_history['val'][best_epoch]['loss'],
            'best_val_accuracy': self.training_history['val'][best_epoch]['accuracy'],
            'best_val_f1': self.training_history['val'][best_epoch]['f1'],
            'final_learning_rate': self.optimizer.param_groups[0]['lr'],
            'model_name': self.model.__class__.__name__
        }

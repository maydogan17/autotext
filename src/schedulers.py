"""
Learning rate schedulers including warmup functionality.

Implements various learning rate scheduling strategies optimized for different model types,
with special emphasis on transformer training with warmup.
"""

import math
from typing import Dict, Any, Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, StepLR, CosineAnnealingLR


class WarmupLRScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup functionality.
    
    Supports linear and cosine warmup followed by various decay strategies.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        warmup_method: str = "linear",
        decay_method: str = "cosine",
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            warmup_method: Warmup method ('linear' or 'cosine')
            decay_method: Decay method after warmup ('cosine', 'linear', 'constant')
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_method = warmup_method
        self.decay_method = decay_method
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            if self.warmup_method == "linear":
                return [base_lr * (self.last_epoch + 1) / self.warmup_steps 
                       for base_lr in self.base_lrs]
            elif self.warmup_method == "cosine":
                return [base_lr * 0.5 * (1 + math.cos(math.pi * (self.warmup_steps - self.last_epoch - 1) / self.warmup_steps))
                       for base_lr in self.base_lrs]
        else:
            # Post-warmup decay phase
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            
            if self.decay_method == "cosine":
                return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                       for base_lr in self.base_lrs]
            elif self.decay_method == "linear":
                return [base_lr * (1 - progress) + self.min_lr * progress
                       for base_lr in self.base_lrs]
            elif self.decay_method == "constant":
                return self.base_lrs
            
        return self.base_lrs
    
    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {
            'last_epoch': self.last_epoch,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'warmup_method': self.warmup_method,
            'decay_method': self.decay_method,
            'min_lr': self.min_lr,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.last_epoch = state_dict['last_epoch']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.warmup_method = state_dict['warmup_method']
        self.decay_method = state_dict['decay_method']
        self.min_lr = state_dict['min_lr']
        self.base_lrs = state_dict['base_lrs']


class AdaptiveLRScheduler:
    """
    Adaptive learning rate scheduler that chooses the best strategy based on model type.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_type: str,
        config: Dict[str, Any],
        total_steps: Optional[int] = None,
        steps_per_epoch: Optional[int] = None
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            model_type: Type of model ('transformer', 'cnn', 'ffn')
            config: Training configuration
            total_steps: Total number of training steps
            steps_per_epoch: Number of steps per epoch
        """
        self.optimizer = optimizer
        self.model_type = model_type
        self.config = config
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        
        # Initialize scheduler based on model type and config
        self.scheduler = self._create_scheduler()
        self.step_counter = 0
        
    def _create_scheduler(self):
        """Create appropriate scheduler based on model type and configuration."""
        lr_config = self.config.get('lr_scheduler', {})
        warmup_config = self.config.get('warmup', {})
        scheduler_type = lr_config.get('type', 'plateau')
        
        # For transformers, prefer warmup + cosine annealing
        if self.model_type == 'transformer' and self.total_steps:
            warmup_steps = warmup_config.get('warmup_steps', 500)
            warmup_ratio = warmup_config.get('warmup_ratio', 0.1)
            
            # Use warmup_ratio if warmup_steps not explicitly set
            if 'warmup_steps' not in warmup_config and 'warmup_ratio' in warmup_config:
                warmup_steps = int(self.total_steps * warmup_ratio)
            
            return WarmupLRScheduler(
                optimizer=self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=self.total_steps,
                warmup_method=warmup_config.get('warmup_method', 'linear'),
                decay_method='cosine',
                min_lr=lr_config.get('eta_min', 1e-6)
            )
        
        # For other models or when total_steps not available
        if scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='max',  # We're maximizing metrics like F1
                factor=lr_config.get('factor', 0.5),
                patience=lr_config.get('patience', 3),
                min_lr=lr_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            return StepLR(
                optimizer=self.optimizer,
                step_size=lr_config.get('step_size', 10),
                gamma=lr_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            T_max = lr_config.get('T_max', self.config.get('max_epochs', 50))
            return CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=T_max,
                eta_min=lr_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'warmup_cosine' and self.total_steps:
            warmup_steps = warmup_config.get('warmup_steps', 500)
            return WarmupLRScheduler(
                optimizer=self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=self.total_steps,
                warmup_method='linear',
                decay_method='cosine',
                min_lr=lr_config.get('eta_min', 1e-6)
            )
        else:
            # No scheduler
            return None
    
    def step(self, metrics: Optional[float] = None, epoch: Optional[int] = None):
        """Step the scheduler."""
        if self.scheduler is None:
            return
            
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metrics is not None:
                self.scheduler.step(metrics)
        elif isinstance(self.scheduler, WarmupLRScheduler):
            self.scheduler.step()
            self.step_counter += 1
        else:
            self.scheduler.step()
    
    def get_last_lr(self):
        """Get the last learning rate."""
        if self.scheduler is None:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        else:
            return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        state = {
            'model_type': self.model_type,
            'config': self.config,
            'total_steps': self.total_steps,
            'steps_per_epoch': self.steps_per_epoch,
            'step_counter': self.step_counter
        }
        if self.scheduler is not None and hasattr(self.scheduler, 'state_dict'):
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.model_type = state_dict['model_type']
        self.config = state_dict['config']
        self.total_steps = state_dict['total_steps']
        self.steps_per_epoch = state_dict['steps_per_epoch']
        self.step_counter = state_dict['step_counter']
        
        if self.scheduler is not None and hasattr(self.scheduler, 'load_state_dict'):
            if 'scheduler_state_dict' in state_dict:
                self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    model_type: str,
    hyperparams: Dict[str, Any],
    config: Dict[str, Any],
    total_steps: Optional[int] = None,
    steps_per_epoch: Optional[int] = None
) -> tuple[torch.optim.Optimizer, Optional[AdaptiveLRScheduler]]:
    """
    Create optimizer and scheduler based on model type and configuration.
    
    Args:
        model: The model to optimize
        model_type: Type of model ('transformer', 'cnn', 'ffn')
        hyperparams: Model-specific hyperparameters
        config: Training configuration
        total_steps: Total number of training steps
        steps_per_epoch: Number of steps per epoch
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Get learning rate (model-specific or default)
    learning_rate = hyperparams.get('learning_rate', config.get('learning_rate', 2e-4))
    
    # Create optimizer
    optimizer_type = config.get('optimizer', 'adamw').lower()
    weight_decay = config.get('weight_decay', 0.01)
    
    if optimizer_type == 'adamw':
        # Separate weight decay for different parameter types (transformers benefit from this)
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    # Create scheduler
    # Enable warmup for transformers by default
    if model_type == 'transformer':
        config = config.copy()
        config['warmup'] = config.get('warmup', {})
        config['warmup']['enabled'] = True
    
    scheduler = AdaptiveLRScheduler(
        optimizer=optimizer,
        model_type=model_type,
        config=config,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch
    )
    
    return optimizer, scheduler

"""
Abstract base class for all text classification models.

This module defines the common interface that all models in the AutoML pipeline
must implement, ensuring consistency across different architectures.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseTextClassifier(nn.Module, ABC):
    """
    Abstract base class for text classification models.
    
    All models (FFN, CNN, Transformer, BERT) must inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        hyperparams: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize the base classifier.
        
        Args:
            vocab_size: Size of the vocabulary
            num_classes: Number of output classes
            hyperparams: Dictionary of hyperparameters from Optuna trial
            device: Device to use for training (cuda/mps/cpu)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.hyperparams = hyperparams
        self.device = device
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def build_model(self) -> None:
        """
        Build the model architecture.
        This method should define all layers and components of the model.
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor (batch_size, sequence_length)
            attention_mask: Optional attention mask for transformer models
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        pass
    
    @abstractmethod
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for Optuna.
        
        Returns:
            Dictionary defining the hyperparameter ranges and types
        """
        pass
    
    def predict(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions with the model.
        
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Predicted class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities
    
    def predict_classes(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Predicted class indices
        """
        probabilities = self.predict(x, attention_mask)
        return torch.argmax(probabilities, dim=-1)
    
    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save the model state and configuration.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'hyperparams': self.hyperparams,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'num_parameters': self.get_num_parameters()
        }, save_path)
    
    @classmethod
    def load_model(cls, load_path: Union[str, Path], device: torch.device):
        """
        Load a saved model.
        
        Args:
            load_path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(load_path, map_location=device)
        
        # Create model instance
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            num_classes=checkpoint['num_classes'],
            hyperparams=checkpoint['hyperparams'],
            device=device
        )
        
        # Build the model architecture
        model.build_model()
        
        # Load the saved weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'num_parameters': self.get_num_parameters(),
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            'hyperparams': self.hyperparams,
            'device': str(self.device)
        }
    
    def freeze_layers(self, freeze_embeddings: bool = True, freeze_encoder: bool = False) -> None:
        """
        Freeze certain layers of the model (useful for transfer learning).
        
        Args:
            freeze_embeddings: Whether to freeze embedding layers
            freeze_encoder: Whether to freeze encoder layers
        """
        # Default implementation - can be overridden by specific models
        for name, param in self.named_parameters():
            if freeze_embeddings and 'embedding' in name.lower():
                param.requires_grad = False
            if freeze_encoder and 'encoder' in name.lower():
                param.requires_grad = False
    
    def unfreeze_all(self) -> None:
        """
        Unfreeze all model parameters.
        """
        for param in self.parameters():
            param.requires_grad = True
    
    def get_layer_names(self) -> List[str]:
        """
        Get names of all layers in the model.
        
        Returns:
            List of layer names
        """
        return [name for name, _ in self.named_modules()]
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.model_name}(vocab_size={self.vocab_size}, num_classes={self.num_classes}, params={self.get_num_parameters():,})"
    
    def __repr__(self) -> str:
        """Detailed representation of the model."""
        return self.__str__()
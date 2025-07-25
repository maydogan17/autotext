"""
Feed-Forward Network (FFN) model for text classification.

A simple but effective neural network that uses word embeddings followed by
fully connected layers for text classification.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseTextClassifier


class FFNTextClassifier(BaseTextClassifier):
    """
    Feed-Forward Network for text classification.
    
    Architecture:
    - Embedding layer
    - Multiple fully connected layers with dropout
    - Global average/max pooling
    - Classification head
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        hyperparams: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize FFN text classifier.
        
        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of output classes
            hyperparams: Hyperparameters from Optuna trial
            device: Device for training
        """
        super().__init__(vocab_size, num_classes, hyperparams, device)
        
        # Extract hyperparameters
        self.embedding_dim = hyperparams['embedding_dim']
        self.hidden_dim = hyperparams['hidden_dim']
        self.num_layers = hyperparams['num_layers']
        self.dropout = hyperparams['dropout']
        self.activation = hyperparams['activation']
        
        # Build model
        self.build_model()
        self.to(device)
    
    def build_model(self) -> None:
        """Build the FFN architecture."""
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )
        
        # Activation function
        if self.activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif self.activation == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            self.activation_fn = nn.ReLU()  # Default
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        
        # First hidden layer (embedding_dim -> hidden_dim)
        self.hidden_layers.append(
            nn.Linear(self.embedding_dim, self.hidden_dim)
        )
        
        # Additional hidden layers (hidden_dim -> hidden_dim)
        for _ in range(self.num_layers - 1):
            self.hidden_layers.append(
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights using best practices."""
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        if self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)
        
        # Initialize linear layer weights
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the FFN model.
        
        Args:
            x: Input token IDs (batch_size, sequence_length)
            attention_mask: Optional attention mask (not used in FFN)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Get embeddings
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Global average pooling to get fixed-size representation
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(embedded).float()
            embedded = embedded * mask_expanded
            pooled = embedded.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple average pooling
            pooled = embedded.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Pass through hidden layers
        hidden = pooled
        for layer in self.hidden_layers:
            hidden = layer(hidden)
            hidden = self.activation_fn(hidden)
            hidden = self.dropout_layer(hidden)
        
        # Classification
        logits = self.classifier(hidden)
        
        return logits
    
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """
        Define hyperparameter search space for Optuna.
        
        Returns:
            Dictionary with parameter names as keys for this model type
        """
        return {
            'embedding_dim': 'categorical',
            'hidden_dim': 'categorical',
            'num_layers': 'categorical', 
            'dropout': 'categorical',
            'activation': 'categorical'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        base_info = super().get_model_info()
        base_info.update({
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'activation': self.activation,
            'architecture': 'Feed-Forward Network'
        })
        return base_info
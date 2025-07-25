"""
Convolutional Neural Network (CNN) model for text classification.

A CNN architecture that uses multiple convolutional filters with different
kernel sizes to capture local patterns in text data.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTextClassifier


class CNNTextClassifier(BaseTextClassifier):
    """
    CNN for text classification.
    
    Architecture:
    - Embedding layer
    - Multiple convolutional layers with different filter sizes
    - Pooling layer (max, avg, or adaptive)
    - Dropout and classification head
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        hyperparams: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize CNN text classifier.
        
        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of output classes
            hyperparams: Hyperparameters from Optuna trial
            device: Device for training
        """
        super().__init__(vocab_size, num_classes, hyperparams, device)
        
        # Extract hyperparameters
        self.embedding_dim = hyperparams['embedding_dim']
        self.num_filters = hyperparams['num_filters']
        self.filter_sizes = hyperparams['filter_sizes']
        self.dropout = hyperparams['dropout']
        self.pooling = hyperparams['pooling']
        
        # Build model
        self.build_model()
        self.to(device)
    
    def build_model(self) -> None:
        """Build the CNN architecture."""
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )
        
        # Convolutional layers
        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=self.num_filters,
                kernel_size=filter_size
            )
            for filter_size in self.filter_sizes
        ])
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Classification head
        # Total features = num_filters * number_of_filter_sizes
        total_features = self.num_filters * len(self.filter_sizes)
        self.classifier = nn.Linear(total_features, self.num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights using best practices."""
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        if self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)
        
        # Initialize convolutional layer weights
        for conv in self.convolutions:
            if isinstance(conv, nn.Conv1d):
                nn.init.xavier_uniform_(conv.weight)
                if conv.bias is not None:
                    nn.init.constant_(conv.bias, 0)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the CNN model.
        
        Args:
            x: Input token IDs (batch_size, sequence_length)
            attention_mask: Optional attention mask for padding
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Get embeddings
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(embedded).float()
            embedded = embedded * mask_expanded
        
        # Transpose for conv1d: (batch_size, embedding_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions and pooling
        conv_outputs = []
        for conv in self.convolutions:
            # Convolution + ReLU
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, conv_seq_len)
            
            # Pooling
            if self.pooling == 'max':
                pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            elif self.pooling == 'avg':
                pooled = F.avg_pool1d(conv_out, kernel_size=conv_out.size(2))
            elif self.pooling == 'adaptive_max':
                pooled = F.adaptive_max_pool1d(conv_out, output_size=1)
            else:
                # Default to max pooling
                pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            
            # Squeeze the last dimension
            pooled = pooled.squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        
        # Apply dropout
        dropped = self.dropout_layer(concatenated)
        
        # Classification
        logits = self.classifier(dropped)
        
        return logits
    
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """
        Define hyperparameter search space for Optuna.
        
        Returns:
            Dictionary with parameter names as keys for this model type
        """
        return {
            'embedding_dim': 'categorical',
            'num_filters': 'categorical',
            'filter_sizes': 'categorical',
            'dropout': 'categorical',
            'pooling': 'categorical'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        base_info = super().get_model_info()
        base_info.update({
            'embedding_dim': self.embedding_dim,
            'num_filters': self.num_filters,
            'filter_sizes': self.filter_sizes,
            'dropout': self.dropout,
            'pooling': self.pooling,
            'architecture': 'Convolutional Neural Network'
        })
        return base_info
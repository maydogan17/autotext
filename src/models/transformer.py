"""
Transformer model for text classification.

A custom transformer implementation with multi-head attention, positional encoding,
and layer normalization for text classification tasks.
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTextClassifier


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, embedding_dim: int, max_position_embeddings: int = 512, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            embedding_dim: Dimension of embeddings
            max_position_embeddings: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_position_embeddings, embedding_dim)
        position = torch.arange(0, max_position_embeddings, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, embedding_dim)
        
        # Register as buffer (not a parameter, but part of module state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, embedding_dim)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(0)
        pe_buffer = getattr(self, 'pe')
        pe = pe_buffer[:seq_len, :].to(x.device)
        x = x + pe
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.out_linear = nn.Linear(embedding_dim, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention scores
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (seq_len, batch_size, embedding_dim)
            key: Key tensor (seq_len, batch_size, embedding_dim)
            value: Value tensor (seq_len, batch_size, embedding_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor after attention
        """
        seq_len, batch_size, embedding_dim = query.size()
        
        # Linear projections and reshape for multi-head attention
        # Goal: (batch_size, num_heads, seq_len, head_dim)
        Q = self.q_linear(query).view(seq_len, batch_size, self.num_heads, self.head_dim)
        K = self.k_linear(key).view(seq_len, batch_size, self.num_heads, self.head_dim)
        V = self.v_linear(value).view(seq_len, batch_size, self.num_heads, self.head_dim)
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.permute(1, 2, 0, 3)  # (seq_len, batch_size, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        K = K.permute(1, 2, 0, 3)
        V = V.permute(1, 2, 0, 3)
        
        # Compute attention
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back to (seq_len, batch_size, embedding_dim)
        attention_output = attention_output.permute(2, 0, 1, 3).contiguous().view(
            seq_len, batch_size, embedding_dim
        )
        
        return self.out_linear(attention_output)
    
    def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q, K, V: Tensors of shape (batch_size, num_heads, seq_len, head_dim)
            mask: Optional mask of shape (batch_size, 1, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        # Compute attention scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # mask should be (batch_size, 1, seq_len, seq_len)
            # Expand to (batch_size, num_heads, seq_len, seq_len)
            if mask.size(1) == 1:
                mask = mask.expand(-1, self.num_heads, -1, -1)
            
            # Apply mask: set masked positions to large negative value
            # Use -1e4 instead of -1e9 to avoid numerical issues
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, V)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, embedding_dim: int, num_heads: int, feedforward_dim: int, dropout: float = 0.1):
        """
        Initialize transformer encoder layer.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            feedforward_dim: Dimension of feedforward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Feedforward network with GELU activation (better for transformers)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),  # GELU works better than ReLU for transformers
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of transformer encoder layer with pre-normalization.
        
        Args:
            x: Input tensor (seq_len, batch_size, embedding_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        # Pre-normalization: better for training stability
        # Self-attention with residual connection and pre-layer norm
        normed_x = self.norm1(x)
        attn_output = self.self_attention(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout(attn_output)
        
        # Feedforward with residual connection and pre-layer norm
        normed_x = self.norm2(x)
        ff_output = self.feedforward(normed_x)
        x = x + self.dropout(ff_output)
        
        return x


class TransformerTextClassifier(BaseTextClassifier):
    """
    Transformer model for text classification.
    
    Architecture:
    - Word embeddings + positional encoding
    - Multiple transformer encoder layers
    - Global average pooling
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
        Initialize transformer text classifier.
        
        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of output classes
            hyperparams: Hyperparameters from Optuna trial
            device: Device for training
        """
        super().__init__(vocab_size, num_classes, hyperparams, device)
        
        # Extract hyperparameters
        self.embedding_dim = hyperparams['embedding_dim']
        self.num_heads = hyperparams['num_heads']
        self.num_layers = hyperparams['num_layers']
        self.feedforward_dim = hyperparams['feedforward_dim']
        self.dropout = hyperparams['dropout']
        self.max_position_embeddings = hyperparams.get('max_position_embeddings', 512)
        
        # Validate that embedding_dim is divisible by num_heads
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})")
        
        # Build model
        self.build_model()
        self.to(device)
    
    def build_model(self) -> None:
        """Build the transformer architecture."""
        # Word embeddings
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dim=self.embedding_dim,
            max_position_embeddings=self.max_position_embeddings,
            dropout=self.dropout
        )
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                feedforward_dim=self.feedforward_dim,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Classification head with layer normalization for better stability
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),  # Add layer norm before classification
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.GELU(),  # Use GELU for consistency
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, self.num_classes)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self) -> None:
        """Initialize model weights with better initialization."""
        # Initialize embeddings with smaller std for better training stability
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.constant_(self.embedding.weight[0], 0)  # padding token
        
        # Initialize linear layers with Xavier/Glorot initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for attention.
        
        Args:
            x: Input tensor (batch_size, seq_len)
            
        Returns:
            Mask tensor (batch_size, 1, seq_len, seq_len)
        """
        batch_size, seq_len = x.size()
        # Create mask where padding tokens (0) are False, others are True
        mask = (x != 0).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
        # Expand to attention matrix size
        mask = mask.expand(batch_size, 1, seq_len, seq_len)
        return mask
    
    def create_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask for transformer layers.
        
        Args:
            attention_mask: Binary mask (batch_size, seq_len) where 1 = valid, 0 = padding
            
        Returns:
            Mask tensor (batch_size, 1, seq_len, seq_len) for multi-head attention
        """
        batch_size, seq_len = attention_mask.size()
        
        # Create mask for attention: (batch_size, seq_len, seq_len)
        # mask[i, j, k] = True if token j can attend to token k
        mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # (batch_size, seq_len, seq_len)
        
        # Add head dimension: (batch_size, 1, seq_len, seq_len)
        mask = mask.unsqueeze(1)
        
        return mask
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        
        Args:
            x: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len) - optional
            
        Returns:
            Classification logits (batch_size, num_classes)
        """
        input_ids = x
        batch_size, seq_len = input_ids.size()
        
        # Word embeddings
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Scale embeddings (common practice in transformers for better training)
        # Use a more conservative scaling factor
        embeddings = embeddings * math.sqrt(self.embedding_dim) * 0.5
        
        # Transpose for positional encoding (seq_len, batch_size, embedding_dim)
        embeddings = embeddings.transpose(0, 1)
        
        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)
        
        # Create attention mask for padding tokens
        # Create mask where True = valid token, False = padding token
        attention_mask_for_layers = None
        if attention_mask is not None:
            # Convert attention mask to the format expected by transformer layers
            # attention_mask: (batch_size, seq_len) where 1 = valid, 0 = padding
            attention_mask_for_layers = self.create_attention_mask(attention_mask)
        else:
            # Create mask from input_ids (assuming 0 is padding token)
            padding_mask = (input_ids != 0)  # (batch_size, seq_len)
            attention_mask_for_layers = self.create_attention_mask(padding_mask)
        
        # Pass through transformer layers with proper masking
        for layer in self.transformer_layers:
            embeddings = layer(embeddings, mask=attention_mask_for_layers)
        
        # Transpose back to (batch_size, seq_len, embedding_dim)
        embeddings = embeddings.transpose(0, 1)
        
        # Global average pooling (respecting padding tokens)
        if attention_mask is not None:
            # Use provided attention mask for proper averaging
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(embeddings).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            avg_embeddings = sum_embeddings / (sum_mask + 1e-9)
        else:
            # Create mask from input_ids and use for averaging
            padding_mask = (input_ids != 0).float()  # (batch_size, seq_len)
            mask_expanded = padding_mask.unsqueeze(-1).expand_as(embeddings)
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            avg_embeddings = sum_embeddings / (sum_mask + 1e-9)
        
        # Classification
        logits = self.classifier(avg_embeddings)
        
        return logits
    
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """
        Define hyperparameter search space for Optuna.
        
        Returns:
            Dictionary with parameter names as keys for this model type
        """
        return {
            'embedding_dim': 'categorical',
            'num_heads': 'categorical',
            'num_layers': 'categorical',
            'feedforward_dim': 'categorical',
            'dropout': 'categorical',
            'max_position_embeddings': 'categorical'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'transformer',
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'feedforward_dim': self.feedforward_dim,
            'dropout': self.dropout,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'max_position_embeddings': self.max_position_embeddings,
            'architecture': 'Custom Transformer'
        }

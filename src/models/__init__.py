"""
Text classification model implementations.

Contains implementations for:
- BaseTextClassifier: Abstract base class
- FFNTextClassifier: Feed-forward network
- CNNTextClassifier: Convolutional neural network
- TransformerTextClassifier: Transformer encoder (to be implemented)
- BERTTextClassifier: Fine-tuned BERT (to be implemented)
"""

from .base import BaseTextClassifier
from .ffn import FFNTextClassifier
from .cnn import CNNTextClassifier

__all__ = [
    "BaseTextClassifier",
    "FFNTextClassifier", 
    "CNNTextClassifier",
]

"""
AutoML Text Classification Pipeline

A simple, configurable AutoML pipeline for text classification that automatically
discovers optimal model architectures and hyperparameters using Optuna.
"""

__version__ = "0.1.0"

# Core components
from .config import Config
from .data_loader import DataProcessor, TextDataset, SimpleTokenizer
from .utils import setup_logger, get_device, seed_everything
from .trainer import Trainer
from .evaluator import ModelEvaluator
from .hpo import HPOOptimizer
from .pipeline import AutoMLPipeline

# Model components
from .models.base import BaseTextClassifier
from .models.ffn import FFNTextClassifier
from .models.cnn import CNNTextClassifier
from .models.transformer import TransformerTextClassifier

__all__ = [
    "Config",
    "DataProcessor", 
    "TextDataset",
    "SimpleTokenizer",
    "setup_logger",
    "get_device", 
    "seed_everything",
    "Trainer",
    "ModelEvaluator",
    "HPOOptimizer",
    "AutoMLPipeline",
    "BaseTextClassifier",
    "FFNTextClassifier",
    "CNNTextClassifier",
    "TransformerTextClassifier",
]

"""
Data loading and preprocessing for the AutoML text classification pipeline.

Handles data loading, tokenization, preprocessing, and dataset creation
for both transformer and non-transformer models.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

import nltk
from nltk.tokenize import word_tokenize

from .config import Config
from .utils import seed_everything


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt')
    except:
        pass

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        pass


class TextDataset(Dataset):
    """Custom PyTorch dataset for text classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int,
        is_transformer: bool = False,
        config_preprocessing: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize text dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            is_transformer: Whether using transformer tokenizer
            config_preprocessing: Preprocessing config for custom tokenizer
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_transformer = is_transformer
        self.config_preprocessing = config_preprocessing or {}
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.is_transformer:
            # Use transformer tokenizer - check if it has the right methods
            if hasattr(self.tokenizer, '__call__'):
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
            else:
                # For custom transformer (not BERT), use simple tokenizer without special tokens
                token_ids = self.tokenizer.encode(text, self.max_length, self.config_preprocessing)
                attention_mask = [1] * len([x for x in token_ids if x != 0])  # Non-padding tokens
                attention_mask.extend([0] * (len(token_ids) - len(attention_mask)))  # Padding tokens
                return {
                    'input_ids': torch.tensor(token_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            # Use custom tokenizer
            token_ids = self.tokenizer.encode(text, self.max_length, self.config_preprocessing)
            attention_mask = [1 if token_id != 0 else 0 for token_id in token_ids]
            
            return {
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long)
            }


class SimpleTokenizer:
    """Simple tokenizer for non-transformer models."""
    
    def __init__(self, vocab_size: int = 20000, min_freq: int = 2):
        """
        Initialize simple tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for token inclusion
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_token = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_built = False
    
    def _preprocess_text(self, text: str, config_preprocessing: Dict[str, Any]) -> str:
        """Apply preprocessing based on config."""
        # Normalize whitespace
        if config_preprocessing.get('normalize_whitespace', True):
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle contractions
        if config_preprocessing.get('handle_contractions', True):
            contractions = {
                "don't": "do not", "won't": "will not", "can't": "cannot",
                "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
                "'d": " would", "'m": " am"
            }
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)
        
        # Convert to lowercase
        if config_preprocessing.get('lowercase', True):
            text = text.lower()
        
        # Remove punctuation if specified
        if config_preprocessing.get('remove_punctuation', False):
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers if specified
        if config_preprocessing.get('remove_numbers', False):
            text = re.sub(r'\d+', '', text)
        
        return text
    
    def build_vocabulary(self, texts: List[str], config_preprocessing: Dict[str, Any]) -> None:
        """Build vocabulary from training texts."""
        token_counts = Counter()
        
        for text in texts:
            # Preprocess text
            processed_text = self._preprocess_text(text, config_preprocessing)
            
            # Tokenize
            tokens = word_tokenize(processed_text)
            
            # Filter by minimum token length
            min_length = config_preprocessing.get('min_token_length', 2)
            tokens = [token for token in tokens if len(token) >= min_length]
            
            token_counts.update(tokens)
        
        # Build vocabulary
        vocab_tokens = [token for token, count in token_counts.most_common() 
                       if count >= self.min_freq]
        
        # Limit vocabulary size
        vocab_tokens = vocab_tokens[:self.vocab_size - 2]  # Reserve space for special tokens
        
        # Add to vocabulary
        for token in vocab_tokens:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        
        self.vocab_built = True
    
    def encode(self, text: str, max_length: int, config_preprocessing: Dict[str, Any]) -> List[int]:
        """Encode text to token IDs with proper preprocessing."""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        # Apply same preprocessing as during vocabulary building
        processed_text = self._preprocess_text(text, config_preprocessing)
        tokens = word_tokenize(processed_text)
        
        # Filter by minimum token length
        min_length = config_preprocessing.get('min_token_length', 2)
        tokens = [token for token in tokens if len(token) >= min_length]
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            token_id = self.token_to_id.get(token, 1)  # 1 is <UNK>
            token_ids.append(token_id)
        
        # Truncate or pad
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([0] * (max_length - len(token_ids)))  # 0 is <PAD>
        
        return token_ids
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_id)


class DataProcessor:
    """Main data processing class."""
    
    def __init__(self, config: Config):
        """
        Initialize data processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.label_encoder = LabelEncoder()
    
    def prepare_data(self, model_type: str) -> Dict[str, Any]:
        """
        Prepare all data for training.
        
        Args:
            model_type: Type of model (ffn, cnn, transformer, bert)
            
        Returns:
            Dictionary containing dataloaders and metadata
        """
        # Set random seed for reproducible splits
        seed_everything(self.config.reproducibility['seed'])
        
        # Load raw data
        train_df, test_df = self._load_data()
        
        # Validate data
        self._validate_data(train_df, test_df)
        
        # Split train data into train/validation
        train_texts, val_texts, train_labels, val_labels = self._split_data(train_df)
        
        # Setup tokenizer
        tokenizer = self.setup_tokenizer(model_type, train_texts)
        
        # Create datasets
        is_transformer = model_type in ['transformer', 'bert']
        max_length = self.config.data['max_length']
        preprocessing_config = self.config.preprocessing if not is_transformer else {}
        
        train_dataset = TextDataset(
            train_texts, train_labels, tokenizer, max_length, is_transformer, preprocessing_config
        )
        val_dataset = TextDataset(
            val_texts, val_labels, tokenizer, max_length, is_transformer, preprocessing_config
        )
        
        # Process test labels  
        test_labels_raw = self.label_encoder.transform(test_df[self.config.data['label_column']])
        test_labels_list = np.array(test_labels_raw).tolist()
        test_dataset = TextDataset(
            test_df[self.config.data['text_column']].tolist(),
            test_labels_list,
            tokenizer, max_length, is_transformer, preprocessing_config
        )
        
        # Create dataloaders
        batch_size = self.config.models['training']['batch_size']
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_labels)
        
        # Get vocabulary size
        if is_transformer:
            vocab_size = 50257  # Default for BERT-like models, will be updated in model init
        else:
            vocab_size = tokenizer.get_vocab_size()
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'tokenizer': tokenizer,
            'num_classes': len(self.label_encoder.classes_),
            'vocab_size': vocab_size,
            'class_weights': class_weights
        }
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test CSV files."""
        dataset_name = self.config.data['dataset_name']
        data_path = Path('data') / dataset_name
        
        train_path = data_path / 'train.csv'
        test_path = data_path / 'test.csv'
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Limit samples if specified with stratified sampling to maintain class distribution
        max_samples = self.config.data.get('max_samples')
        if max_samples is not None and max_samples < len(train_df):
            label_col = self.config.data['label_column']
            
            # Get class distribution
            class_counts = train_df[label_col].value_counts().sort_index()
            total_classes = len(class_counts)
            
            # Calculate samples per class (equal distribution)
            samples_per_class = max_samples // total_classes
            remaining_samples = max_samples % total_classes
            
            # Stratified sampling to maintain equal class distribution
            sampled_dfs = []
            for i, (class_label, class_count) in enumerate(class_counts.items()):
                class_df = train_df[train_df[label_col] == class_label]
                
                # Add one extra sample to first 'remaining_samples' classes
                target_samples = samples_per_class + (1 if i < remaining_samples else 0)
                
                # Sample from this class (or take all if fewer samples available)
                if len(class_df) >= target_samples:
                    sampled_class_df = class_df.sample(
                        n=target_samples, 
                        random_state=self.config.data['random_state']
                    )
                else:
                    sampled_class_df = class_df  # Take all available samples
                
                sampled_dfs.append(sampled_class_df)
            
            # Combine all sampled classes and shuffle
            train_df = pd.concat(sampled_dfs, ignore_index=True)
            train_df = train_df.sample(
                frac=1, 
                random_state=self.config.data['random_state']
            ).reset_index(drop=True)
        
        return train_df, test_df
    
    def _validate_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Validate that required columns exist."""
        text_col = self.config.data['text_column']
        label_col = self.config.data['label_column']
        
        for df_name, df in [('train', train_df), ('test', test_df)]:
            if text_col not in df.columns:
                raise ValueError(f"Text column '{text_col}' not found in {df_name} data")
            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found in {df_name} data")
    
    def _split_data(self, train_df: pd.DataFrame) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Split training data into train/validation sets."""
        text_col = self.config.data['text_column']
        label_col = self.config.data['label_column']
        
        texts = train_df[text_col].tolist()
        labels = train_df[label_col].tolist()
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        val_size = self.config.data['validation_size']
        random_state = self.config.data['random_state']
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, encoded_labels, test_size=val_size, 
            random_state=random_state, stratify=encoded_labels
        )
        
        return train_texts, val_texts, train_labels.tolist(), val_labels.tolist()
    
    def setup_tokenizer(self, model_type: str, train_texts: List[str]) -> Any:
        """Setup appropriate tokenizer based on model type."""
        if model_type == 'bert':
            # Use transformer tokenizer
            model_name = 'bert-base-uncased'  # Default, will be overridden by hyperparams
            if AutoTokenizer is not None:
                return AutoTokenizer.from_pretrained(model_name)
            else:
                raise ImportError("transformers library is required for BERT models")
        else:
            # Build custom tokenizer
            preprocessing_config = self.config.preprocessing
            vocab_size = preprocessing_config['vocab_size']
            min_freq = preprocessing_config['min_token_freq']
            
            tokenizer = SimpleTokenizer(vocab_size=vocab_size, min_freq=min_freq)
            tokenizer.build_vocabulary(train_texts, preprocessing_config)
            
            return tokenizer
    
    def calculate_class_weights(self, labels: List[int]) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        class_weights_strategy = self.config.training.get('class_weights', 'balanced')
        
        if class_weights_strategy == 'balanced':
            unique_labels = np.array(list(set(labels)))
            weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_labels,
                y=np.array(labels)
            )
            return torch.tensor(weights, dtype=torch.float32)
        else:
            # Return equal weights
            num_classes = len(set(labels))
            return torch.ones(num_classes, dtype=torch.float32)
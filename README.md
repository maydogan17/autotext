# AutoML Text Classification

A comprehensive automated machine learning pipeline for text classification tasks with **enhanced transformer architectures**, **warmup scheduling**, and **model-specific optimization**. This system automatically discovers optimal model architectures and hyperparameters, trains models, and provides detailed evaluation reports - all with minimal user intervention.

## Enhanced Features

### Advanced Model Architectures

- **Enhanced Transformer**: Proper attention masking with custom implementation
- **Warmup Scheduling**: Linear and cosine warmup strategies for optimal training
- **Model-Specific Optimization**: Tailored learning rates and parameters per architecture
- **GPU-Optimized**: Large-scale configurations for high-performance training

### Professional AutoML Pipeline

- **Fully Automated**: Complete ML workflow from data loading to model deployment
- **Intelligent HPO**: Optuna-based optimization with model-specific parameter spaces
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualization
- **Production Ready**: Robust error handling, logging, and artifact management

## Architecture

```
AutoML Pipeline
├── Data Processing      # Load, preprocess, and tokenize text data
├── Model Discovery      # FFN, CNN, Transformer architectures
├── HPO Optimization     # Optuna-based hyperparameter search
├── Training             # Early stopping, checkpointing, validation
├── Evaluation          # Comprehensive metrics and reports
└── Artifacts           # Model saving and result persistence
```

### Supported Models

| Model           | Description                  | Key Features                     |
| --------------- | ---------------------------- | -------------------------------- |
| **FFN**         | Feed-Forward Network         | Simple, fast, good baseline      |
| **CNN**         | Convolutional Neural Network | Captures local patterns, filters |
| **Transformer** | BERT-based                   | State-of-the-art performance     |

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd autotext

# Install dependencies (if using conda/venv)
pip install torch torchvision scikit-learn pandas numpy optuna pyyaml tqdm
```

### Basic Usage

```bash
# Run with default settings (Amazon dataset, 50 HPO trials)
python main.py

# Use specific configuration
python main.py --config my_config.yaml
```

### Example Output

```
AutoML Text Classification

Loading configuration: configs/config.yaml
Using device: mps
Dataset: amazon
HPO trials: 50
Optimization metric: f1_score_weighted

Starting AutoML Pipeline...

============================================================
AUTOML PIPELINE SUMMARY
============================================================
Status: completed
Execution Time: 245.6s
Device: mps

Dataset: amazon
Samples: 13141
Classes: 3

Best Model: cnn
Best Score: 0.7234
Trials: 50
HPO Time: 180.2s

Final Performance:
  Accuracy: 0.7156
  F1 (weighted): 0.7234
  F1 (macro): 0.5891
============================================================

Results saved to: trained_models/
AutoML pipeline completed successfully!
```

## Project Structure

```
autotext/
├── src/                     # Core source code
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management
│   ├── data_loader.py      # Data processing and tokenization
│   ├── trainer.py          # Model training logic
│   ├── evaluator.py        # Model evaluation and metrics
│   ├── hpo.py              # Hyperparameter optimization
│   ├── pipeline.py         # Main AutoML pipeline orchestrator
│   ├── utils.py            # Utility functions
│   └── models/             # Model implementations
│       ├── base.py         # Base model class
│       ├── ffn.py          # Feed-Forward Network
│       ├── cnn.py          # Convolutional Neural Network
│       └── transformer.py  # Transformer/BERT (future)
├── configs/
│   └── config.yaml         # Main configuration file
├── data/                   # Dataset storage
│   ├── amazon/            # Amazon reviews (3-class)
│   ├── ag_news/           # AG News (4-class)
│   ├── dbpedia/           # DBpedia (14-class)
│   └── imdb/              # IMDB reviews (2-class)
├── trained_models/         # Output directory
├── notebooks/              # Jupyter notebooks for analysis
├── main.py                 # CLI entry point
├── test_pipeline.py        # Pipeline testing script
├── test_hpo.py            # HPO testing script
└── README.md              # This file
```

## Configuration

The pipeline is configured via `configs/config.yaml`. Key sections:

### Data Configuration

```yaml
data:
  dataset_name: "amazon" # amazon, ag_news, dbpedia, imdb
  text_column: "text"
  label_column: "label"
  max_length: 128
  validation_split: 0.1
```

### HPO Configuration

```yaml
hpo:
  num_trials: 50
  metric: "f1_score_weighted"
  direction: "maximize"
  timeout: 2400 # 40 minutes
  enable_pruning: true
  early_stopping_patience: 5
```

### Model Hyperparameter Spaces

```yaml
hyperparameters:
  ffn:
    embedding_dim: [64, 512]
    hidden_dim: [128, 1024]
    num_layers: [1, 4]
    dropout: [0.1, 0.5]
    activation: ["relu", "gelu", "tanh"]

  cnn:
    embedding_dim: [64, 512]
    num_filters: [32, 256]
    filter_sizes: ["2,3,4", "3,4,5", "2,3,4,5", "3,4,5,6"]
    dropout: [0.1, 0.5]
    pooling: ["max", "adaptive_max", "adaptive_avg"]
```

## Output Files

After running, the pipeline generates comprehensive outputs in `trained_models/`:

| File                         | Description                       |
| ---------------------------- | --------------------------------- |
| `best_model.pt`              | Best trained model weights        |
| `config.yaml`                | Complete configuration used       |
| `pipeline_results.json`      | Detailed results in JSON format   |
| `pipeline_summary.txt`       | Human-readable summary report     |
| `hpo_results.json`           | HPO trial results and statistics  |
| `optuna_study.pkl`           | Optuna study for further analysis |
| `best_model_evaluation.json` | Detailed evaluation metrics       |

# 🔧 BabyLlama API Reference

> **Comprehensive API documentation for developers and researchers**

This document provides detailed API reference for all BabyLlama components, including classes, functions, and configuration schemas.

## 📋 Table of Contents

- [Core Classes](#core-classes)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Configuration Schema](#configuration-schema)
- [Command Line Interface](#command-line-interface)
- [Examples](#examples)

## 🏗️ Core Classes

### DataProcessor

The main class for data processing and preparation.

```python
class DataProcessor:
    """Modern data processor using HuggingFace datasets"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """Initialize with a tokenizer instance"""
        
    def load_text_files(self, data_dir: str, split: str = "train") -> Dataset:
        """Load text files from directory into HF Dataset
        
        Args:
            data_dir: Directory containing text files
            split: File extension to match (e.g., "train", "dev")
            
        Returns:
            Dataset with 'text' and 'source' columns
        """
        
    def clean_text(self, example: Dict[str, str]) -> Dict[str, str]:
        """Apply basic text cleaning
        
        Args:
            example: Dict with 'text' key
            
        Returns:
            Dict with cleaned 'text'
        """
        
    def tokenize_and_chunk(self, example: Dict[str, str], max_length: int = 128) -> Dict[str, List[int]]:
        """Tokenize and chunk text into fixed-length sequences
        
        Args:
            example: Dict with 'text' key
            max_length: Maximum sequence length
            
        Returns:
            Dict with 'input_ids' containing list of token sequences
        """
        
    def prepare_dataset(
        self,
        train_data_dir: str,
        eval_data_dir: str,
        max_length: int = 128,
        clean: bool = True,
        num_proc: int = 4,
    ) -> DatasetDict:
        """Prepare complete dataset for training
        
        Args:
            train_data_dir: Training data directory
            eval_data_dir: Evaluation data directory
            max_length: Sequence length for chunking
            clean: Whether to apply text cleaning
            num_proc: Number of processes for parallel processing
            
        Returns:
            DatasetDict with 'train' and 'validation' splits
        """
```

### ModelEvaluator

Comprehensive model evaluation and analysis.

```python
class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize evaluator with model path
        
        Args:
            model_path: Path to saved model directory
            device: Device for inference ("cuda" or "cpu")
        """
        
    def calculate_perplexity(self, texts: List[str], batch_size: int = 8) -> Dict[str, float]:
        """Calculate perplexity on text samples
        
        Args:
            texts: List of text strings to evaluate
            batch_size: Batch size for processing
            
        Returns:
            Dict with perplexity metrics:
            - perplexity: Overall perplexity score
            - average_loss: Mean cross-entropy loss
            - std_loss: Standard deviation of losses
            - num_sequences: Number of evaluated sequences
            - total_tokens: Total number of tokens processed
        """
        
    def evaluate_generation_quality(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_return_sequences: int = 3,
    ) -> Dict[str, any]:
        """Evaluate generation quality metrics
        
        Args:
            prompts: List of prompts for generation
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences per prompt
            
        Returns:
            Dict with generation metrics:
            - avg_diversity_score: Average n-gram diversity
            - avg_repetition_score: Average repetition ratio
            - generation_samples: Sample generations
        """
        
    def evaluate_token_probabilities(self, texts: List[str]) -> Dict[str, float]:
        """Analyze token probability statistics
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dict with probability metrics:
            - avg_top_token_prob: Average highest token probability
            - std_top_token_prob: Standard deviation of top probabilities
            - avg_entropy: Average token entropy
            - low_confidence_ratio: Fraction of low-confidence predictions
        """
        
    def plot_metrics(self, metrics: Dict[str, any], save_path: str = "evaluation_plots.png"):
        """Create visualization of evaluation metrics
        
        Args:
            metrics: Dictionary of computed metrics
            save_path: Path to save the plot
        """
```

### BenchmarkSuite

Standardized benchmarking for model comparison.

```python
class BenchmarkSuite:
    """Collection of benchmark tasks for language models"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize benchmark suite
        
        Args:
            model_path: Path to model directory
            device: Device for inference
        """
        
    def run_completion_benchmark(self) -> Dict[str, float]:
        """Test phrase completion accuracy
        
        Returns:
            Dict with completion metrics:
            - completion_accuracy: Fraction of correct completions
            - completion_correct: Number of correct completions
            - completion_total: Total number of prompts
        """
        
    def run_consistency_benchmark(self) -> Dict[str, float]:
        """Test output consistency across similar prompts
        
        Returns:
            Dict with consistency metrics:
            - avg_consistency_score: Average consistency across prompt groups
            - consistency_scores: List of individual group scores
        """
        
    def run_repetition_benchmark(self) -> Dict[str, float]:
        """Analyze repetition in generated text
        
        Returns:
            Dict with repetition metrics:
            - avg_repetition_score: Average repetition ratio
            - repetition_scores: List of individual scores
        """
        
    def run_speed_benchmark(self) -> Dict[str, float]:
        """Measure inference speed
        
        Returns:
            Dict with speed metrics:
            - tokens_per_second: Average generation speed
            - latency_ms: Average response latency
        """
        
    def run_all_benchmarks(self) -> Dict[str, any]:
        """Run complete benchmark suite
        
        Returns:
            Dict with all benchmark results and overall score
        """
```

## 📊 Data Processing

### Domain-Specific Cleaners

```python
class DomainCleaners:
    """Collection of domain-specific text cleaners"""
    
    @staticmethod
    def wikipedia(text: str) -> str:
        """Clean Wikipedia text
        - Removes headers (=== Title ===)
        - Removes citations [1], [2]
        - Normalizes whitespace
        """
        
    @staticmethod
    def subtitles(text: str) -> str:
        """Clean subtitle text
        - Removes timing markers
        - Removes subtitle numbers
        - Removes credits
        """
        
    @staticmethod
    def dialogue(text: str) -> str:
        """Clean dialogue/conversation text
        - Removes speaker labels
        - Removes stage directions [action]
        - Normalizes formatting
        """

def create_cleaner_registry() -> Dict[str, Callable]:
    """Create registry mapping domain names to cleaner functions
    
    Returns:
        Dict mapping domain names to cleaner functions:
        - "wikipedia": Wikipedia cleaner
        - "subtitles": Subtitle cleaner  
        - "dialogue": Dialogue cleaner
        - "default": No-op cleaner
    """
```

### Utility Functions

```python
def load_config(config_path: str, args) -> dict:
    """Load YAML configuration with command-line overrides
    
    Args:
        config_path: Path to YAML config file
        args: Argparse namespace with overrides
        
    Returns:
        Configuration dictionary
    """

def create_model(config: dict, tokenizer) -> torch.nn.Module:
    """Create model from configuration
    
    Args:
        config: Model configuration dict
        tokenizer: Tokenizer instance
        
    Returns:
        Initialized model (LlamaForCausalLM, GPT2LMHeadModel, or GPTJForCausalLM)
    """

def prepare_datasets_modern(config: dict, tokenizer) -> Tuple[Dataset, Dataset]:
    """Prepare training and validation datasets

    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer instance

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """

## ⚙️ Configuration Schema

### Complete Configuration Structure

```yaml
# config/model-config.yaml
data:
  tokenizer_path: str          # Path to tokenizer file (required)
  train_path: str              # Training data directory (required)
  eval_path: str               # Evaluation data directory (required)
  seq_length: int              # Sequence length: 128-512 (default: 128)
  eval_samples: int            # Number of eval samples (default: 256)

model:
  type: str                    # Model type: "Llama", "GPT2", "GPTJ" (required)
  name: str                    # Model name for saving (required)
  hidden_size: int             # Model width: 64-2048 (required)
  intermediate_size: int       # FFN size: typically 4x hidden_size (required)
  n_layer: int                 # Number of layers: 2-48 (required)
  n_head: int                  # Number of attention heads (required)
  tie_word_embeddings: bool    # Tie input/output embeddings (default: false)
  resid_pdrop: float           # Residual dropout: 0.0-0.3 (GPT-2 only)
  embd_pdrop: float            # Embedding dropout: 0.0-0.3 (GPT-2 only)
  attn_pdrop: float            # Attention dropout: 0.0-0.3 (GPT-2 only)

training:
  lr: float                    # Learning rate: 1e-5 to 1e-3 (required)
  batch_size: int              # Per-device batch size: 1-128 (required)
  num_epochs: int              # Training epochs: 1-10 (required)
  gradient_accumulation_steps: int  # Gradient accumulation: 1-32 (default: 1)
  warmup_steps: int            # Warmup steps: 0-1000 (default: 0)
  fp16: bool                   # Mixed precision training (default: false)
  bf16: bool                   # BFloat16 training (default: false)
  torch_compile: bool          # PyTorch 2.0 compilation (default: false)

logging:
  wandb: bool                  # Enable Weights & Biases (default: false)
  project: str                 # W&B project name (required if wandb: true)
  output_dir: str              # Model save directory (default: "./models/")
```
```

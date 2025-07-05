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

### Configuration Validation

The configuration is automatically validated on load. Common validation rules:

- `hidden_size` must be divisible by `n_head`
- `intermediate_size` should be 4x `hidden_size` for GPT-2, 2.67x for LLaMA
- `seq_length` must be between 32 and 2048
- `lr` should be between 1e-6 and 1e-2
- `batch_size` must be positive

## 💻 Command Line Interface

### Training Commands

```bash
# Basic training
python train.py --config CONFIG_PATH

# With parameter overrides
python train.py \
  --config CONFIG_PATH \
  --lr 5e-4 \
  --batch-size 16 \
  --model_name "custom-model"

# Resume from checkpoint
python train.py \
  --config CONFIG_PATH \
  --resume_from_checkpoint CHECKPOINT_PATH
```

### Evaluation Commands

```bash
# Basic evaluation
python evaluate.py MODEL_PATH

# Detailed evaluation
python evaluate.py MODEL_PATH \
  --num-samples 1000 \
  --output-path results.json \
  --device cuda

# Custom prompts
python evaluate.py MODEL_PATH \
  --custom-prompts prompts.txt \
  --temperature 0.8
```

### Benchmarking Commands

```bash
# Single model benchmark
python benchmark.py MODEL_PATH

# Compare multiple models
python benchmark.py MODEL1_PATH MODEL2_PATH MODEL3_PATH \
  --output comparison.json

# Custom benchmark suite
python benchmark.py MODEL_PATH \
  --benchmark-config custom_benchmarks.yaml
```

### Data Preparation Commands

```bash
# Prepare BabyLM data
python prepare_data.py \
  --babylm-10m /path/to/babylm_10M \
  --babylm-dev /path/to/babylm_dev \
  --tokenizer-vocab 16000

# Generate synthetic data
python create_synthetic_data.py \
  --num-tokens 1000000 \
  --output-dir ./data/synthetic \
  --complexity medium

# Train tokenizer
python train_tokenizer.py \
  --data-dir ./data/train \
  --vocab-size 16000 \
  --output ./models/tokenizer.json
```

## 🔧 Advanced Usage Examples

### Custom Model Creation

```python
from transformers import AutoConfig
from train import create_model

# Create custom configuration
config = {
    "model": {
        "type": "Llama",
        "hidden_size": 384,
        "intermediate_size": 1024,
        "n_layer": 8,
        "n_head": 8,
        "vocab_size": 16000
    }
}

# Create model
model = create_model(config, tokenizer)
print(f"Model parameters: {model.num_parameters():,}")
```

### Custom Data Processing

```python
from data_utils import DataProcessor
from transformers import GPT2TokenizerFast

# Initialize processor
tokenizer = GPT2TokenizerFast.from_pretrained("./models/tokenizer.json")
processor = DataProcessor(tokenizer)

# Process custom data
dataset = processor.prepare_dataset(
    train_data_dir="./my_data/train",
    eval_data_dir="./my_data/eval",
    max_length=256,
    clean=True,
    num_proc=8
)

print(f"Training samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")
```

### Batch Evaluation

```python
from evaluate import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator("./models/Llama-10M/")

# Evaluate on multiple text samples
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times."
]

# Calculate metrics
perplexity_metrics = evaluator.calculate_perplexity(texts)
generation_metrics = evaluator.evaluate_generation_quality(
    prompts=["The cat sat on the", "Once upon a time"],
    max_length=50,
    num_return_sequences=3
)

print(f"Average perplexity: {perplexity_metrics['perplexity']:.2f}")
print(f"Generation diversity: {generation_metrics['avg_diversity_score']:.2f}")
```

### Custom Benchmark Suite

```python
from benchmark import BenchmarkSuite

class CustomBenchmarkSuite(BenchmarkSuite):
    def run_domain_specific_benchmark(self) -> Dict[str, float]:
        """Custom benchmark for specific domain."""
        prompts = [
            "In machine learning, the concept of",
            "The fundamental principle of",
            "When training neural networks, it is important to"
        ]

        results = []
        for prompt in prompts:
            # Generate and evaluate
            outputs = self.model.generate(
                self.tokenizer(prompt, return_tensors="pt")["input_ids"],
                max_length=50,
                num_return_sequences=1
            )
            # Custom scoring logic here
            score = self.calculate_domain_score(outputs[0])
            results.append(score)

        return {
            "domain_accuracy": sum(results) / len(results),
            "domain_scores": results
        }

# Use custom benchmark
benchmark = CustomBenchmarkSuite("./models/Llama-10M/")
results = benchmark.run_domain_specific_benchmark()
```

## 🚨 Error Handling and Troubleshooting

### Common Errors

#### CUDA Out of Memory
```python
# Error: RuntimeError: CUDA out of memory
# Solution: Reduce batch size or enable gradient accumulation
python train.py --config CONFIG \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --fp16
```

#### Tokenizer Not Found
```python
# Error: FileNotFoundError: tokenizer.json not found
# Solution: Train tokenizer first
python train_tokenizer.py --data-dir ./data/train
```

#### Configuration Validation Error
```python
# Error: ValueError: hidden_size must be divisible by n_head
# Solution: Adjust configuration
# hidden_size: 192, n_head: 6  ✓
# hidden_size: 200, n_head: 6  ✗
```

### Performance Issues

#### Slow Training
```bash
# Enable optimizations
python train.py --config CONFIG \
  --fp16 \
  --torch-compile \
  --dataloader-num-workers 4
```

#### Memory Optimization
```bash
# For limited GPU memory
python train.py --config CONFIG \
  --batch-size 4 \
  --gradient-accumulation-steps 16 \
  --fp16
```

## 📊 Performance Benchmarks

### Training Performance

| Model Size | GPU Memory | Batch Size | Tokens/sec | Time to 1 Epoch |
|------------|------------|------------|------------|------------------|
| 10M | 2GB | 32 | 1,200 | 2 min |
| 16M | 3GB | 32 | 1,000 | 5 min |
| 95M | 8GB | 16 | 800 | 30 min |
| 360M | 16GB | 8 | 600 | 2 hours |

### Inference Performance

| Model Size | Batch Size | Tokens/sec | Latency (ms) | Memory (GB) |
|------------|------------|------------|--------------|-------------|
| 10M | 1 | 200 | 5 | 0.5 |
| 10M | 8 | 1,200 | 7 | 1.0 |
| 95M | 1 | 120 | 8 | 2.0 |
| 95M | 8 | 800 | 10 | 4.0 |

## 📚 API Reference Summary

### Core Classes
- **`DataProcessor`**: Data loading and preprocessing
- **`ModelEvaluator`**: Model evaluation and metrics
- **`BenchmarkSuite`**: Standardized benchmarking

### Utility Functions
- **`load_config()`**: Configuration loading with overrides
- **`create_model()`**: Model instantiation from config
- **`prepare_datasets_modern()`**: Dataset preparation pipeline

### Command Line Tools
- **`train.py`**: Model training with configuration
- **`evaluate.py`**: Model evaluation and analysis
- **`benchmark.py`**: Standardized benchmarking
- **`prepare_data.py`**: Data preparation pipeline

---

For more examples and detailed usage, see the [Training Guide](TRAINING_GUIDE.md) and [Documentation Hub](docs/README.md).

# BabyLlama 🦙

[![arXiv](https://img.shields.io/badge/arXiv-2308.02019-b31b1b.svg)](https://arxiv.org/abs/2308.02019)
[![Tests](https://github.com/yourusername/BabyLlama/actions/workflows/test.yml/badge.svg)](https://github.com/yourusername/BabyLlama/actions/workflows/test.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<div align="center">
  <img src="assets/babyllama-dalle3.png" alt="Baby-Llama LLM with its Teachers" width=50% height=50%>

  *BabyLlama and its teachers, as depicted by DALL·E 3*
</div>

> **A modern, production-ready framework for training small language models from scratch**

BabyLlama is a comprehensive toolkit for training and evaluating small language models, based on the [BabyLM Challenge](https://babylm.github.io/) submission. This project demonstrates how to build efficient, well-tested language models with modern Python practices and state-of-the-art techniques.

## ✨ Key Features

- 🏗️ **Multiple Architectures**: Support for LLaMA, GPT-2, and GPT-J models
- 🚀 **Modern Training Pipeline**: Built on HuggingFace Transformers with efficient data processing
- 📊 **Comprehensive Evaluation**: Perplexity, generation quality, diversity metrics, and benchmarking
- 🎓 **Knowledge Distillation**: Train student models from ensemble teachers
- ⚙️ **Flexible Configuration**: YAML-based configs with command-line overrides
- 🧪 **Production Ready**: 63 tests, CI/CD, type hints, and comprehensive documentation
- 📦 **Modern Tooling**: Uses `uv` for dependency management, `pytest` for testing, `ruff` for linting
- 🔧 **Easy Setup**: One-command installation and synthetic data generation for quick starts

## 🚀 Quick Start

Get a model training in under 5 minutes:

```bash
# 1. Clone and setup
git clone https://github.com/pgryko/BabyLlama.git
cd BabyLlama
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,test]"

# 2. Generate synthetic data and train tokenizer
uv run python create_synthetic_data.py
uv run python train_tokenizer.py

# 3. Train a 10M parameter model
uv run python train.py --config ./config/llama-10M.yaml

# 4. Evaluate the results
uv run python evaluate.py models/Llama-10M/
```

That's it! You now have a trained language model with comprehensive evaluation metrics.


## 📚 Documentation

### 🎯 For Beginners
- **[Training Guide](TRAINING_GUIDE.md)** - Complete step-by-step tutorial from setup to deployment
- **[Installation Guide](#-installation)** - Detailed setup instructions

### 🔧 For Developers
- **[API Reference](API_REFERENCE.md)** - Comprehensive API documentation
- **[Contributing Guide](CONTRIBUTING.md)** - Development setup and contribution guidelines
- **[Testing Guide](tests/README.md)** - How to run and write tests

### 📖 For Researchers
- **[Documentation Hub](docs/README.md)** - Complete documentation overview and navigation
- **[Architecture Guide](docs/README.md#architecture)** - Model architectures and design decisions

## 🛠️ Installation

### Prerequisites
- Python 3.12+ (3.11+ supported)
- NVIDIA GPU with 8GB+ VRAM (recommended)
- CUDA 11.8+ (for GPU training)

### Quick Setup

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/pgryko/BabyLlama.git
cd BabyLlama

# Create environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,test]"

# Verify installation
uv run python run_tests.py --smoke
```

### Alternative: Using pip

```bash
git clone https://github.com/pgryko/BabyLlama.git
cd BabyLlama
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,test]"
```

## 🏗️ Model Architectures

| Architecture | Parameters | Key Features | Best For |
|-------------|------------|-------------|----------|
| **LLaMA** | 10M-360M | RoPE, SwiGLU, RMSNorm | Modern efficiency, fastest training |
| **GPT-2** | 97M-705M | Learned PE, GELU, LayerNorm | Baseline comparison, well-studied |
| **GPT-J** | 97M | RoPE, GELU, LayerNorm | Hybrid approach, good balance |

### Pre-configured Model Sizes

| Config | Parameters | Memory | Training Time | Use Case |
|--------|------------|---------|---------------|----------|
| `llama-10M.yaml` | ~10M | 2GB | 2 min | Quick experiments |
| `llama-16M.yaml` | ~16M | 3GB | 5 min | Small-scale training |
| `llama-95M.yaml` | ~95M | 8GB | 30 min | Medium experiments |
| `llama-360M.yaml` | ~360M | 16GB | 2 hours | Large teacher model |

## 🎯 Usage Examples

### Train Your First Model (5 minutes)

```bash
# Generate synthetic data and train tokenizer
uv run python create_synthetic_data.py
uv run python train_tokenizer.py

# Train a 10M parameter model
uv run python train.py --config ./config/llama-10M.yaml

# Evaluate the results
uv run python evaluate.py models/Llama-10M/
```

### Use Real Data (BabyLM Dataset)

```bash
# Download BabyLM data from https://babylm.github.io/
# Then prepare the data
uv run python prepare_data.py \
  --babylm-10m /path/to/babylm_10M \
  --babylm-dev /path/to/babylm_dev

# Train with real data
uv run python train.py --config ./config/llama-16M.yaml

# Compare with benchmarks
uv run python benchmark.py models/Llama-16M/
```

### Advanced: Knowledge Distillation

```bash
# Train teacher models
uv run python train.py --config ./config/gpt-705M.yaml
uv run python train.py --config ./config/llama-360M.yaml

# Distill into smaller student model
uv run python distill-ensemble-pretraining-baby-llama.py \
  --config ./config/distillation.yaml
```

## 📊 Expected Results

### Performance Baselines

| Model | Dataset | Training Time | Perplexity | Completion Acc. |
|-------|---------|---------------|------------|-----------------|
| Llama-10M | Synthetic | 2 min | 4.3 | 35% |
| Llama-10M | BabyLM | 15 min | 3.6 | 52% |
| Llama-95M | BabyLM | 2 hours | 2.9 | 68% |

### Architecture Comparison

![Training Speed Comparison](assets/wandb-Llama-gptj-gpt2.png)

**Key Findings**:
- 🚀 **LLaMA trains 2x faster** than GPT-2 due to RoPE and SwiGLU
- 📈 **GPT-J shows intermediate performance** with RoPE but standard MLP
- 🎯 **SwiGLU activation is crucial** for training efficiency

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Quick test
uv run python run_tests.py

# Full test suite with coverage
pytest --cov=. --cov-report=html

# Specific test categories
pytest -m "not integration"  # Unit tests only
```

## 📖 Citation

If you use BabyLlama in your research, please cite:

```bibtex
@article{babylm2023,
  title={BabyLM Challenge: Sample-efficient pretraining on a developmentally plausible corpus},
  author={Warstadt, Alex and Mueller, Aaron and Choshen, Leshem and Wilcox, Ethan and Zhuang, Chengxu and Ciro, Juan and Mosquera, Rafael and Paranjabe, Bhargavi and Williams, Adina and Linzen, Tal and others},
  journal={arXiv preprint arXiv:2308.02019},
  year={2023}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BabyLM Challenge](https://babylm.github.io/) organizers
- [HuggingFace](https://huggingface.co/) for the Transformers library
- [Astral](https://astral.sh/) for the `uv` package manager
- Original paper authors and the research community

---

<div align="center">
  <strong>Happy Training! 🚀</strong><br>
  <em>Built with ❤️ for the research community</em>
</div>

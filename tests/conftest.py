import sys
from pathlib import Path
import pytest
import torch
import tempfile
import shutil
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_text_files(temp_dir):
    """Create sample text files for testing."""
    texts = [
        "This is a sample text file for testing purposes.",
        "Another file with some content.\nMultiple lines here.",
        "Third file with special characters: @#$%^&*()",
    ]

    files = []
    for i, text in enumerate(texts):
        file_path = temp_dir / f"test_{i}.txt"
        file_path.write_text(text)
        files.append(file_path)

    return files


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample configuration file."""
    config_content = """
model_type: llama
model_size: tiny
vocab_size: 1000
hidden_size: 64
num_hidden_layers: 2
num_attention_heads: 2
intermediate_size: 256
max_position_embeddings: 128
hidden_act: gelu
layer_norm_epsilon: 1e-6
initializer_range: 0.02
use_cache: true
tie_word_embeddings: true
rope_theta: 10000.0

# Training configuration
learning_rate: 1e-3
batch_size: 4
num_epochs: 1
gradient_accumulation_steps: 1
warmup_steps: 100
weight_decay: 0.01
max_grad_norm: 1.0
seed: 42
"""
    config_path = temp_dir / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = Mock(return_value="decoded text")
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.vocab_size = 1000
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.config = Mock()
    model.config.vocab_size = 1000
    model.config.hidden_size = 64
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)

    # Mock forward pass
    logits = torch.randn(1, 10, 1000)
    model.forward = Mock(return_value=Mock(logits=logits))

    return model


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_torch_seed():
    """Reset PyTorch random seed before each test."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

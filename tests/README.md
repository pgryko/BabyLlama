# 🧪 BabyLlama Test Suite

Comprehensive test suite ensuring code quality and reliability across all BabyLlama components.

## 📁 Test Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── test_data_utils.py       # Data processing and utilities (21 tests)
├── test_train.py            # Training pipeline and models (15 tests)
├── test_evaluate.py         # Evaluation metrics and analysis (14 tests)
├── test_integration.py      # End-to-end workflows (5 tests)
└── README.md               # This file
```

## 🚀 Quick Start

### Install Dependencies
```bash
# Using uv (recommended)
uv pip install -e ".[test]"

# Using pip
pip install -e ".[test]"
```

### Run Tests
```bash
# Quick smoke test
uv run python run_tests.py --smoke

# All tests
pytest

# With coverage report
pytest --cov=. --cov-report=html

# Specific test file
pytest tests/test_data_utils.py

# Parallel execution
pytest -n auto
```

## 🏷️ Test Categories

### Unit Tests (`pytest -m "not integration"`)
Test individual functions and methods in isolation:

- **Data Processing**: Text cleaning, tokenization, chunking
- **Model Creation**: Configuration loading, model instantiation
- **Evaluation Metrics**: Perplexity, diversity, repetition scores
- **Utilities**: Helper functions and domain cleaners

### Integration Tests (`pytest -m integration`)
Test complete workflows and component interactions:

- **Training Pipeline**: End-to-end model training
- **Evaluation Workflow**: Model loading and evaluation
- **Data Pipeline**: Data preparation and processing
- **Cross-Component**: Integration between modules

### Performance Tests
Validate speed and memory usage:

- **Training Speed**: Tokens per second benchmarks
- **Memory Usage**: GPU and CPU memory consumption
- **Inference Speed**: Generation performance
- **Data Loading**: Dataset processing efficiency

## 📊 Test Coverage

Current coverage: **95%+** across all modules

| Module | Coverage | Tests | Focus Areas |
|--------|----------|-------|-------------|
| `data_utils.py` | 98% | 21 | Data processing, cleaning, tokenization |
| `train.py` | 94% | 15 | Model training, configuration, checkpoints |
| `evaluate.py` | 96% | 14 | Metrics calculation, model evaluation |
| `benchmark.py` | 92% | 8 | Standardized benchmarks, comparisons |
| Integration | 90% | 5 | End-to-end workflows |

### Coverage Report
```bash
# Generate HTML coverage report
pytest --cov=. --cov-report=html

# View report
open htmlcov/index.html
```

## ✍️ Writing Tests

### Test Naming Convention
```python
def test_function_name_expected_behavior():
    """Test description of what is being tested."""
    # Test implementation

def test_function_name_edge_case():
    """Test edge case or error condition."""
    # Test implementation

@pytest.mark.integration
def test_end_to_end_workflow():
    """Test complete workflow integration."""
    # Integration test implementation
```

### Example Unit Test
```python
import pytest
from data_utils import DataProcessor

def test_data_processor_tokenize_basic():
    """Test basic tokenization functionality."""
    processor = DataProcessor(mock_tokenizer)

    example = {"text": "Hello world"}
    result = processor.tokenize_and_chunk(example, max_length=10)

    assert "input_ids" in result
    assert len(result["input_ids"]) > 0
    assert all(len(seq) <= 10 for seq in result["input_ids"])

def test_data_processor_empty_text():
    """Test handling of empty text input."""
    processor = DataProcessor(mock_tokenizer)

    with pytest.raises(ValueError, match="Text cannot be empty"):
        processor.tokenize_and_chunk({"text": ""})
```

### Example Integration Test
```python
@pytest.mark.integration
def test_complete_training_pipeline():
    """Test end-to-end training workflow."""
    # Setup test configuration
    config = create_test_config()

    # Run training
    model = train_model(config)

    # Verify model was created and saved
    assert model is not None
    assert os.path.exists(config["output_dir"])

    # Test evaluation
    metrics = evaluate_model(config["output_dir"])
    assert "perplexity" in metrics
    assert metrics["perplexity"] > 0
```

### Test Fixtures
```python
# conftest.py
@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.vocab_size = 1000
    return tokenizer

@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "model": {"type": "Llama", "hidden_size": 64},
        "training": {"lr": 1e-4, "batch_size": 2}
    }
```

## 🔧 Test Configuration

### Pytest Configuration (`pytest.ini`)
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    integration: marks tests as integration tests
    slow: marks tests as slow running
    gpu: marks tests that require GPU
addopts =
    --strict-markers
    --disable-warnings
    -v
```

### Running Specific Test Types
```bash
# Unit tests only (fast)
pytest -m "not integration"

# Integration tests only
pytest -m integration

# GPU tests (if GPU available)
pytest -m gpu

# Slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"
```

## 🚀 Continuous Integration

### GitHub Actions Workflow
Tests run automatically on:
- ✅ Every push to `main`/`develop` branches
- ✅ All pull requests
- ✅ Multiple platforms: Ubuntu, macOS, Windows
- ✅ Python versions: 3.11, 3.12
- ✅ With and without GPU

### CI Test Matrix
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ["3.11", "3.12"]
    include:
      - os: ubuntu-latest
        python-version: "3.12"
        gpu: true
```

## 🐛 Debugging Tests

### Running Tests with Debug Output
```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Debug specific test
pytest tests/test_data_utils.py::test_specific_function -v -s
```

### Common Test Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Import errors** | `ModuleNotFoundError` | Install test dependencies: `uv pip install -e ".[test]"` |
| **Fixture not found** | `fixture 'name' not found` | Check `conftest.py` or import fixtures |
| **GPU tests failing** | CUDA errors | Skip GPU tests on CPU-only systems: `pytest -m "not gpu"` |
| **Slow tests** | Tests timeout | Run with more time: `pytest --timeout=300` |

## 📈 Test Metrics

### Performance Benchmarks
- **Unit tests**: < 0.1s per test
- **Integration tests**: < 30s per test
- **Full suite**: < 5 minutes
- **Coverage collection**: < 10 minutes

### Quality Gates
- ✅ All tests must pass
- ✅ Coverage must be > 90%
- ✅ No critical security issues
- ✅ Code style compliance

## 🤝 Contributing Tests

When adding new features:

1. **Write tests first** (TDD approach)
2. **Include edge cases** and error conditions
3. **Add integration tests** for new workflows
4. **Update documentation** if test behavior changes
5. **Ensure CI passes** before submitting PR

### Test Review Checklist
- [ ] Tests cover new functionality
- [ ] Edge cases and errors handled
- [ ] Integration tests for workflows
- [ ] Performance impact considered
- [ ] Documentation updated

---

**Happy Testing! 🧪** Well-tested code is reliable code.
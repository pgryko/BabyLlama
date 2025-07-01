# BabyLlama Test Suite

This directory contains the comprehensive test suite for the BabyLlama project.

## Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_data_utils.py` - Unit tests for data processing utilities
- `test_train.py` - Unit tests for training components
- `test_evaluate.py` - Unit tests for evaluation metrics
- `test_integration.py` - Integration tests for end-to-end workflows

## Running Tests

### Install test dependencies
```bash
pip install -e ".[test]"
```

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=. --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_data_utils.py
```

### Run tests in parallel
```bash
pytest -n auto
```

### Run only unit tests
```bash
pytest -m unit
```

### Run only integration tests
```bash
pytest -m integration
```

## Test Coverage

The test suite aims for high coverage of critical components:

- **Data Processing**: Text cleaning, tokenization, chunking, domain-specific cleaners
- **Model Training**: Config loading, model creation, dataset preparation
- **Evaluation**: Perplexity calculation, generation metrics, diversity/repetition scores
- **Integration**: End-to-end pipelines, cross-component interactions

## Writing New Tests

When adding new features, please include corresponding tests:

1. Unit tests for individual functions/methods
2. Integration tests for feature interactions
3. Edge case testing for error conditions
4. Performance tests for critical paths

## CI/CD

Tests are automatically run on:
- Every push to main/develop branches
- All pull requests
- Multiple OS platforms (Ubuntu, macOS, Windows)
- Python 3.11 and 3.12

See `.github/workflows/test.yml` for CI configuration.
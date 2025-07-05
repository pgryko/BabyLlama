# 🤝 Contributing to BabyLlama

We welcome contributions! This guide will help you get started with contributing to BabyLlama.

## 🚀 Quick Start for Contributors

### Development Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/yourusername/BabyLlama.git
cd BabyLlama

# 3. Set up development environment
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,test]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify setup
uv run python run_tests.py
```

## 📋 Types of Contributions

### 🐛 Bug Reports
- Use GitHub Issues with the "bug" label
- Include minimal reproduction steps
- Provide system information (Python version, GPU, etc.)
- Include error messages and stack traces

### 💡 Feature Requests
- Use GitHub Issues with the "enhancement" label
- Describe the use case and expected behavior
- Consider implementation complexity and scope

### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements

## 🛠️ Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

Follow our coding standards:

#### Code Quality Standards

- **Formatting**: Use `black` for code formatting
- **Linting**: Use `ruff` for linting and import sorting
- **Type Hints**: Add type hints to all functions
- **Docstrings**: Use Google-style docstrings
- **Testing**: Add tests for new functionality

#### Example Code Style

```python
def process_text(
    text: str, 
    max_length: int = 128, 
    clean: bool = True
) -> List[str]:
    """Process text into chunks for training.
    
    Args:
        text: Input text to process
        max_length: Maximum sequence length
        clean: Whether to apply text cleaning
        
    Returns:
        List of processed text chunks
        
    Raises:
        ValueError: If text is empty or max_length is invalid
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    
    # Implementation here
    return processed_chunks
```

### 3. Run Tests

```bash
# Run all tests
uv run python run_tests.py

# Run specific test categories
pytest -m "not integration"  # Unit tests only
pytest tests/test_data_utils.py  # Specific module

# Run with coverage
pytest --cov=. --cov-report=html
```

### 4. Update Documentation

- Update docstrings for new/modified functions
- Update README.md if adding new features
- Add examples to TRAINING_GUIDE.md if relevant
- Update API_REFERENCE.md for new APIs

### 5. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: add support for custom attention mechanisms

- Implement CustomAttention class
- Add configuration options for attention type
- Update model creation to support custom attention
- Add tests for new attention mechanism"
```

#### Commit Message Format

Use conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions/modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `style:` - Code style changes

### 6. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Include:
# - Clear description of changes
# - Link to related issues
# - Screenshots/examples if applicable
# - Test results
```

## 🧪 Testing Guidelines

### Writing Tests

```python
import pytest
from your_module import your_function

def test_your_function_basic_case():
    """Test basic functionality."""
    result = your_function("input")
    assert result == "expected_output"

def test_your_function_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError, match="Invalid input"):
        your_function("")

@pytest.mark.integration
def test_end_to_end_workflow():
    """Test complete workflow integration."""
    # Integration test implementation
    pass
```

### Test Categories

- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test speed and memory usage
- **Edge Case Tests**: Test error conditions

## 📚 Documentation Standards

### Docstring Format

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """Brief description of what the function does.
    
    Longer description if needed. Explain the purpose,
    algorithm, or important implementation details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter with default
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer
        
    Example:
        >>> result = example_function("hello", 5)
        >>> print(result)
        True
    """
```

### README Updates

When adding features, update relevant sections:
- Quick Start examples
- Configuration options
- API examples
- Performance benchmarks

## 🎯 Areas for Contribution

### High Priority
- 🚀 **Performance Optimization**: Training and inference speed improvements
- 🧠 **New Model Architectures**: Support for additional transformer variants
- 📊 **Evaluation Metrics**: Additional benchmarks and evaluation methods
- 🔧 **Tooling**: Better debugging, profiling, and monitoring tools

### Medium Priority
- 📚 **Documentation**: Tutorials, examples, and guides
- 🧪 **Testing**: Increase test coverage and add edge cases
- 🎨 **User Experience**: CLI improvements and better error messages
- 🔄 **CI/CD**: Workflow improvements and automation

### Good First Issues
- 🐛 **Bug Fixes**: Small, well-defined issues
- 📝 **Documentation**: Typo fixes and clarifications
- 🧹 **Code Cleanup**: Refactoring and style improvements
- ✅ **Test Coverage**: Adding tests for existing functionality

## 🔍 Code Review Process

### What We Look For
- **Correctness**: Does the code work as intended?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Style**: Does it follow our coding standards?
- **Performance**: Are there any performance implications?
- **Compatibility**: Does it work across supported platforms?

### Review Timeline
- Initial review within 2-3 days
- Follow-up reviews within 1-2 days
- Merge after approval and CI passes

## 🆘 Getting Help

### Community Resources
- **GitHub Discussions**: Questions and community support
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and references

### Contact
- Open an issue for bugs or feature requests
- Use discussions for questions and help
- Tag maintainers for urgent issues

## 📄 License

By contributing to BabyLlama, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to BabyLlama! 🚀**

Your contributions help make language model training more accessible to researchers and practitioners worldwide.

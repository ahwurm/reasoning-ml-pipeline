# Contributing to Binary Reasoning ML Pipeline

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## How to Contribute

### Reporting Issues

1. **Check existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

### Suggesting Enhancements

1. **Open a discussion** first for major changes
2. **Provide context**:
   - Why is this enhancement needed?
   - How would it benefit users?
   - Any potential drawbacks?

### Contributing Code

#### 1. Fork and Clone
```bash
# Fork on GitHub, then:
git clone https://github.com/your-username/reasoning-ml-pipeline.git
cd reasoning-ml-pipeline
git remote add upstream https://github.com/original/reasoning-ml-pipeline.git
```

#### 2. Create a Branch
```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or fixes

#### 3. Set Up Development Environment
```bash
# Use pyenv for Python version management
pyenv install 3.10.11
pyenv virtualenv 3.10.11 reasoning-dev
pyenv shell reasoning-dev

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

#### 4. Make Your Changes

Follow these guidelines:

##### Code Style
- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 88 characters (Black default)
- Use descriptive variable names

##### Documentation
- Add docstrings to all functions/classes
- Update README if adding new features
- Include examples in docstrings

Example docstring:
```python
def predict_binary_reasoning(
    prompt: str,
    model_name: str = "logistic_regression"
) -> Tuple[str, float]:
    """
    Make a binary prediction for a mathematical reasoning question.
    
    Args:
        prompt: The yes/no question to answer
        model_name: Name of the model to use
        
    Returns:
        Tuple of (prediction, confidence) where prediction is "yes"/"no"
        and confidence is a float between 0 and 1
        
    Example:
        >>> predict_binary_reasoning("Is 25 a prime number?")
        ("no", 0.98)
    """
```

##### Testing
- Write tests for new functionality
- Maintain test coverage above 80%
- Use pytest for testing

```python
# tests/test_new_feature.py
def test_new_feature():
    """Test description."""
    # Arrange
    input_data = prepare_test_data()
    
    # Act
    result = new_feature(input_data)
    
    # Assert
    assert result.is_valid()
    assert result.accuracy > 0.9
```

#### 5. Run Quality Checks
```bash
# Format code
black src tests
isort src tests

# Run linters
flake8 src tests
mypy src

# Run tests
pytest
pytest --cov=src --cov-report=html

# Check documentation
pydocstyle src
```

#### 6. Commit Your Changes
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add uncertainty calibration for Bayesian models"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test changes
- `chore:` - Build/auxiliary changes

#### 7. Push and Create Pull Request
```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference any related issues
- Screenshots/examples if applicable
- Confirmation that tests pass

### Pull Request Guidelines

1. **Keep PRs focused** - One feature/fix per PR
2. **Update tests** - Add/modify tests as needed
3. **Update documentation** - Keep docs in sync
4. **Pass CI checks** - All tests and linters must pass
5. **Request review** - Tag relevant maintainers

### Code Review Process

1. **Automated checks** run first (tests, linting)
2. **Maintainer review** for code quality and design
3. **Address feedback** promptly and professionally
4. **Approval and merge** once all checks pass

## Development Guidelines

### Adding New Models

1. Create model class in `src/models/`
2. Implement required interface:
```python
class NewModel:
    def __init__(self, config: dict):
        """Initialize model with configuration."""
        
    def train(self, X, y) -> dict:
        """Train model and return metrics."""
        
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        
    def save(self, path: str):
        """Save model to disk."""
        
    def load(self, path: str):
        """Load model from disk."""
```

3. Add training script in `src/`
4. Update model registry
5. Add tests and documentation

### Adding New Features

1. **Discuss first** - Open an issue or discussion
2. **Design document** - For complex features
3. **Incremental development** - Small, reviewable PRs
4. **Feature flags** - For experimental features

### Performance Considerations

- Profile code for bottlenecks
- Use appropriate data structures
- Consider memory usage
- Document complexity (O notation)

### Security Guidelines

- Never commit secrets or API keys
- Validate all user inputs
- Use parameterized queries
- Keep dependencies updated

## Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── fixtures/       # Test data
└── conftest.py     # Pytest configuration
```

### Writing Tests
- Test happy path and edge cases
- Use descriptive test names
- Keep tests independent
- Mock external dependencies

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test
pytest -k "test_logistic_regression_accuracy"
```

## Documentation Standards

### Code Documentation
- Every public function needs a docstring
- Include type hints
- Provide usage examples
- Document exceptions raised

### Project Documentation
- Keep README.md updated
- Update CHANGELOG.md
- Add to relevant docs/ files
- Include diagrams where helpful

## Release Process

1. **Version bump** following semantic versioning
2. **Update CHANGELOG.md** with release notes
3. **Create release PR** with all changes
4. **Tag release** after merge
5. **Deploy** to production

## Getting Help

- 📚 Read existing documentation
- 🔍 Search existing issues
- 💬 Ask in discussions
- 📧 Contact maintainers

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to make this project better! 🎉
# Contributing to Supreme System V5

Thank you for your interest in Supreme System V5!

## Commercial Project Notice

This is a **commercial, proprietary project**. All contributions are subject to the commercial license agreement.

## Before Contributing

### For External Contributors

**Note:** This project is not open-source. Contributions are accepted on a case-by-case basis and require:

1. **Signed Contributor License Agreement (CLA)**
2. **Written approval from project maintainer**
3. **Agreement to proprietary license terms**

### For Internal Team Members

Follow the development guidelines below.

## Development Process

### 1. Branch Naming

```
feature/description    # New features
bugfix/description     # Bug fixes
hotfix/description     # Critical fixes
docs/description       # Documentation
test/description       # Test additions
```

### 2. Commit Messages

Follow semantic commit format:

```
feat: Add new trading strategy
fix: Resolve memory leak in data pipeline
docs: Update API documentation
test: Add tests for risk management
chore: Update dependencies
```

### 3. Code Quality

#### Required Before Commit:

- ✅ All tests passing
- ✅ Code coverage maintained or improved
- ✅ Linting passes (flake8, black)
- ✅ Type hints added
- ✅ Docstrings complete (Google style)

#### Commands:

```
# Run tests
pytest tests/ -v

# Check coverage
pytest --cov=src --cov-report=term-missing

# Lint
flake8 src/ tests/
black src/ tests/ --check

# Type check
mypy src/
```

### 4. Testing Requirements

- **Unit tests**: Required for all new code
- **Integration tests**: Required for feature interactions
- **Edge cases**: Must be covered
- **Documentation**: Tests must be documented

### 5. Documentation

#### Required Documentation:

1. **Docstrings**: All public functions/classes
2. **README updates**: For new features
3. **CHANGELOG**: For all changes
4. **API docs**: For public interfaces
5. **Examples**: For new features

#### Documentation Style:

```
def function_name(arg1: Type, arg2: Type) -> ReturnType:
    """
    Brief description.
    
    Longer description with details about functionality.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When invalid input
    
    Example:
        >>> function_name(1, 2)
        3
    """
```

### 6. Pull Request Process

1. **Create feature branch** from `main`
2. **Implement changes** with tests
3. **Update documentation**
4. **Run full test suite**
5. **Submit PR** with description
6. **Address review comments**
7. **Merge** after approval

#### PR Template:

```
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Test addition

## Testing
- [ ] All tests pass
- [ ] Coverage maintained/improved
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] No breaking changes
```

## Code Style

### Python

- **Formatter**: Black (line length 88)
- **Linter**: flake8
- **Type checker**: mypy
- **Docstring**: Google style

### Naming Conventions

- **Classes**: PascalCase
- **Functions**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Private**: _leading_underscore

## Testing Guidelines

### Test Structure

```
class TestFeatureName:
    """Test suite for FeatureName"""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data"""
        return data
    
    def test_basic_functionality(self, setup_data):
        """Test 1: Basic case"""
        assert result == expected
    
    def test_edge_case(self):
        """Test 2: Edge case"""
        with pytest.raises(ValueError):
            invalid_operation()
```

### Coverage Requirements

- **Overall**: Maintain 27%+ (increasing to 70%+)
- **New code**: 80%+ coverage required
- **Critical modules**: 90%+ coverage (risk management)

## Review Process

### Code Review Checklist

- [ ] Code quality and style
- [ ] Test coverage adequate
- [ ] Documentation complete
- [ ] No security issues
- [ ] Performance considered
- [ ] Backward compatibility

### Review Timeline

- **Initial review**: Within 48 hours
- **Follow-up**: Within 24 hours
- **Final approval**: Maintainer discretion

## Questions?

For questions about contributing:

1. Check existing documentation
2. Search closed issues
3. Open a discussion (not issue)
4. Contact maintainer if needed

## Legal

By contributing, you agree that your contributions will be licensed under the project's commercial license.

---

**Last Updated:** November 17, 2025  
**Version:** 1.0

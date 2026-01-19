# Contributing

## Development Workflow

1. Fork the repository
2. Create a feature branch from `main`
3. Make changes following commit conventions
4. Run tests: `uv run pytest`
5. Push and submit a pull request

## Git Commit Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change, no new feature or fix |
| `perf` | Performance improvement |
| `test` | Adding/correcting tests |
| `chore` | Build process or auxiliary tools |

### Scopes

- `model` - Model architecture
- `train` - Training scripts
- `data` - Dataset handling
- `test` - Test files
- `docs` - Documentation

### Examples

```bash
feat(model): add SimpleGAT baseline

fix(train): correct learning rate scheduler

docs: update README with CLI examples
```

## Code Style

- Use `ruff` for linting and formatting
- Follow PEP 8 with 88 character line limit
- Use type hints for function signatures
- Write docstrings for public functions

```bash
# Format code
make format

# Check lint
make lint
```

## Testing

- Place tests next to source files (Go-style)
- Name test files as `*_test.py`
- Run from project root: `uv run pytest`

```
models/
  simple_gcn.py
  simple_gcn_test.py  # Test file next to source
```

## Python Imports

- Use relative imports for intra-package
- Use absolute imports for external packages

```python
import torch                      # Third-party: absolute
from .simple_gcn import SimpleGCN # Same package: relative
from ..shared import metrics      # Parent package: relative
```

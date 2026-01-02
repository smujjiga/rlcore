# rlcore Project Setup

## Project Structure

```
rlcore/
├── rlcore/                    # Python package (all .py files go here)
│   ├── __init__.py           # Package initialization
│   ├── main.py               # Main entry point
│   └── example.py            # Example module
├── pyproject.toml            # Project configuration
├── ruff.toml                 # Ruff linter/formatter config
├── rlcore.code-workspace     # VS Code workspace settings
├── uv.lock                   # Dependency lock file
└── .venv/                    # Virtual environment
```

## Technology Stack

- **Python**: 3.12.3
- **Package Manager**: UV (modern, fast Python package manager)
- **Linter/Formatter**: Ruff (fast Python linter and formatter)
- **Editor**: VS Code with Ruff extension

## Development Setup

### Initial Setup

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Open the workspace in VS Code:
   ```bash
   code rlcore.code-workspace
   ```

3. Install recommended VS Code extensions:
   - Ruff (`charliermarsh.ruff`)
   - Python (`ms-python.python`)
   - Pylance (`ms-python.vscode-pylance`)

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a development dependency
uv add --dev package-name
```

### Code Quality

```bash
# Run linter
uv run ruff check rlcore/

# Auto-fix linting issues
uv run ruff check --fix rlcore/

# Format code
uv run ruff format rlcore/

# Check formatting without changes
uv run ruff format --check rlcore/
```

### Running the Package

```bash
# Run the main module
python -m rlcore.main

# Or after installing
uv run rlcore
```

## Ruff Configuration

Configured in `ruff.toml`:
- Line length: 88 characters
- Target: Python 3.12
- Enabled rules: pycodestyle, pyflakes, isort, pyupgrade, flake8-bugbear, flake8-comprehensions, flake8-simplify

## VS Code Features

When using `rlcore.code-workspace`:
- **Format on save** - Automatically formats Python files
- **Auto-fix on save** - Fixes linting issues automatically
- **Organize imports on save** - Sorts and organizes imports
- **Default formatter** - Ruff is set as the default Python formatter

## Best Practices

1. All Python source files go in the `rlcore/` directory
2. Run `ruff check` before committing
3. Use type hints for better code quality
4. Keep functions focused and well-documented
5. Write tests alongside your code

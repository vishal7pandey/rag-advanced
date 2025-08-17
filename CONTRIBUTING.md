# Contributing to RAG Advanced

Thank you for your interest in contributing!

## Development Environment

- Python 3.11+
- Create a virtual environment
- Install project + dev tools:
  ```bash
  pip install -e .[dev]
  ```

## Coding Standards

- Style: ruff (PEP 8 rules + default ruff rules)
- Types: mypy on `app/`
- Tests: pytest, aim for meaningful unit tests and fast runtime
- Keep modules small and cohesive. Favor composition and clear interfaces.

## Pre-commit Checks

Run all checks locally before opening a PR:

```bash
scripts/dev_check.sh    # macOS/Linux
scripts/dev_check.ps1   # Windows
pytest --cov=app --cov-report=term-missing
```

## Branching Strategy

- `main`: stable branch
- feature branches: `feat/<short-name>`
- fixes: `fix/<short-name>`
- docs: `docs/<short-name>`

Open a PR against `main` when ready. Keep PRs focused and small if possible.

## Commit Messages

- Use clear, imperative subject lines (e.g., "Add hybrid RRF retriever")
- Reference issues where helpful

## Tests

- Place tests in `tests/`
- Use monkeypatch/fakes to avoid heavy downloads or network calls
- Ensure new features have coverage and existing behavior stays green

## Releases

- Update `CHANGELOG.md`
- Bump version in `pyproject.toml`
- Tag a release (e.g., `v1.0.0`)

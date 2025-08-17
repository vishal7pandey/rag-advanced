$ErrorActionPreference = "Stop"
pytest --cov=app --cov-report=term-missing
ruff check .
mypy app

#!/usr/bin/env bash
set -euo pipefail

pytest -q
ruff check .
mypy app

.PHONY: setup lint type test cov check run uv-setup uv-test uv-cov uv-check uv-run
setup:
	pip install -e .[dev]
lint:
	ruff check .
type:
	mypy app
test:
	pytest -q
cov:
	pytest --cov=app --cov-report=term-missing
check: lint type test
run:
	streamlit run app/ui/streamlit_app.py

uv-setup:
	uv venv && uv pip install -e .[dev]
uv-test:
	uv run pytest -q
uv-cov:
	uv run pytest --cov=app --cov-report=term-missing
uv-check:
	uv run ruff check . && uv run mypy app
uv-run:
	uv run streamlit run app/ui/streamlit_app.py

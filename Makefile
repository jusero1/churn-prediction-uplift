.PHONY: setup data train lint format test check

setup:  ## Install dependencies and git hooks
	uv sync --dev
	uv run pre-commit install

data:  ## Download Cell2Cell (Kaggle) and Hillstrom (direct) + checksums
	uv run python -m churn_uplift.data.download

train:  ## Rebuild models, metrics and figures from the raw data
	uv run python -m churn_uplift.train

lint:  ## Static checks: ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-format the codebase
	uv run ruff format .
	uv run ruff check --fix .

test:  ## Run the test suite with coverage
	uv run pytest --cov=churn_uplift --cov-report=term-missing

check: lint test  ## Run before every commit

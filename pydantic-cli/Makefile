PYTHONPATH=src

.PHONY: setup
setup:
	@# Setup Project
	uv init pydantic-cli

	@# Install Python
	uv python install
	uv python list
	uv python pin 3.13.2

	@# Install Dependencies
	uv add typer
	uv add flake8 black isort mypy pytest taskipy --dev

.PHONY: re-install
re-install:
	#uv sync
	rm -rf .venv
	uv venv .venv
	uv pip install -e .

.PHONY: activate-env
activate-env:
	source .venv/bin/activate

.PHONY: lint
lint:
	uvx ruff format src
	uvx ruff check --fix src
	# uvx isort src
	# uvx black src
	# uvx flake8 src

.PHONY: lint2
lint2:
	uv run task flake8


.PHONY: run
run:
	python -m src.cli.main simple-agent
	python -m src.cli.main agent-structured-response
	python -m src.cli.main agent-structured-dependencies
	python -m src.cli.main agent-with-tools
	python -m src.cli.main agent-with-self-correction

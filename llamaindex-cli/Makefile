PYTHONPATH=src

###############################################################################
# Setup
###############################################################################
.PHONY: setup
setup:
	@# Setup Project
	uv init llamaindex-cli

	@# Install Python
	uv python install
	uv python list
	uv python pin 3.13.2

	@# Install Dependencies
	uv add typer llama-index-core llama-index llama-index-llms-lmstudio
	uv add flake8 black isort mypy pytest taskipy --dev

.PHONY: re-install
re-install:
	#uv sync
	rm -rf uv.lock
	rm -rf .venv
	uv venv .venv
	uv pip install -e .

###############################################################################
# Development
###############################################################################
# sourceコマンドはなぜかMakefileで動かない
# make: source: No such file or directory
# またvenvをactivate後すぐにpythonコマンドを実行してもmoduleの解決ができないことがある
.PHONY: activate
activate:
	source .venv/bin/activate

.PHONY: deactivate
deactivate:
	deactivate

.PHONY: lint
lint:
	uvx ruff format src
	uvx ruff check --fix src

.PHONY: lint2
lint2:
	uvx isort src
	uvx black src
	uv run task flake8

###############################################################################
# Execution
###############################################################################
.PHONY: run-example
run-example:
	PYTHONPATH=src uv run -m src.cli.main docs-agent
	PYTHONPATH=src uv run -m src.cli.main docs-agent --storage text

# Use local LLM API server
# Note: run `lms load` first
.PHONY: run-with-local-llm
run-with-local-llm:
	PYTHONPATH=src uv run -m src.cli.main --local docs-agent
	PYTHONPATH=src uv run -m src.cli.main --local docs-agent --storage text

bug-pattern:
	PYTHONPATH=src uv run -m src.cli.main --local local-llm


###############################################################################
# Utilities
###############################################################################
# `.venv`を除いたディレクトリ内の`__pycache__`を削除する
.PHONY: clean-cache
clean-cache:
	find . -type d -name '.venv' -prune -o -type d -name '__pycache__' -exec rm -r {} + -print
	rm -rf ./src/llamaindex-cli.egg-info


###############################################################################
# Local LLM API Server
# - use LM Studio
###############################################################################
# After running LM Studio
.PHONY: load-lms
load-lms:
	lms load llama-3.2-3b-instruct --identifier "llama3"
	#lms load text-embedding-nomic-embed-text-v1.5-embedding --identifier "text-embedding"
	lms load text-embedding-nomic-embed-text-v1.5-embedding --identifier "text-embedding-ada-002"
	@#lms load multilingual-e5-large-mlx --identifier "multilingual-e5"
	

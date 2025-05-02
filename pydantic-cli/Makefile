PYTHONPATH=src

###############################################################################
# Setup
###############################################################################
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
	rm -rf uv.lock
	rm -rf .venv
	uv venv .venv
	uv pip install -e .

.PHONY: update-deps
update-deps:
	uv sync --upgrade
	uv lock

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
	# Type Checking
	#uvx mypy src
	uv run mypy src

.PHONY: lint2
lint2:
	# Formatting
	uvx isort src
	uvx black src
	# Type Checking
	uvx mypy src
	# Linting
	uv run task flake8

###############################################################################
# Execution
###############################################################################

.PHONY: run
run:
	uv run -m src.cli.main simple-agent
	uv run -m src.cli.main agent-structured-response
	uv run -m src.cli.main agent-structured-dependencies
	uv run -m src.cli.main agent-with-tools
	uv run -m src.cli.main agent-with-self-correction

###############################################################################
# Utilities
###############################################################################

# `.venv`を除いたディレクトリ内の`__pycache__`を削除する
.PHONY: clean-cache
clean-cache:
	find . -type d -name '.venv' -prune -o -type d -name '__pycache__' -exec rm -r {} + -print
	rm -rf ./src/pydantic-cli.egg-info


###############################################################################
# Local LLM API Server
# - use LM Studio
###############################################################################
# After running LM Studio
.PHONY: load-lms
load-lms:
	lms load llama-3.2-3b-instruct --identifier "llama3.2"
	lms load text-embedding-nomic-embed-text-v1.5-embedding --identifier "text-embedding"
	
###############################################################################
# Ollama
###############################################################################

# ollama pull qwen3:30b-a3b
# ollama pull qwen3:8b

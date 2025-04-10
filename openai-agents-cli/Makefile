PYTHONPATH=src

###############################################################################
# Setup
###############################################################################
.PHONY: setup
setup:
	@# Setup Project
	uv init openai-agents-cli

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
	uvx mypy src

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
.PHONY: run-example
run-example:
	PYTHONPATH=src uv run -m src.cli.main query-tech-guide --question "What is an advantage of using Python?"
	PYTHONPATH=src uv run -m src.cli.main query-tech-guide-chat --question "What is an advantage of using Rust?"
	PYTHONPATH=src uv run -m src.cli.main news-agent
	PYTHONPATH=src uv run -m src.cli.main embedding
	PYTHONPATH=src uv run -m src.cli.main search-similarity --id 1
	
# Use local LLM API server
# Note: run `lms load` first
.PHONY: run-with-local-llm
run-with-local-llm:
	PYTHONPATH=src uv run -m src.cli.main --local query-tech-guide-chat --question "What is an advantage of using Rust?"

# WIP
.PHONY: run-vector
run-vector:
	# PYTHONPATH=src uv run -m src.cli.main query --question "What is an advantage of using Python?"
	# PYTHONPATH=src uv run -m src.cli.main query --question "What types of curry are there?"
	# PYTHONPATH=src uv run -m src.cli.main query --question "Give me the recipe for chicken curry"
	# PYTHONPATH=src uv run -m src.cli.main query --question "What is the largest river in Japan?"
	# PYTHONPATH=src uv run -m src.cli.main query --question "How far is it between Tokyo and Osaka?"
	

###############################################################################
# Utilities
###############################################################################
# `.venv`を除いたディレクトリ内の`__pycache__`を削除する
.PHONY: clean-cache
clean-cache:
	find . -type d -name '.venv' -prune -o -type d -name '__pycache__' -exec rm -r {} + -print
	rm -rf ./src/openai_agents_cli.egg-info


###############################################################################
# Local LLM API Server
# - use LM Studio
###############################################################################
# After running LM Studio
.PHONY: load-lms
load-lms:
	lms load llama-3.2-3b-instruct --identifier "llama3"
	lms load multilingual-e5-large-mlx --identifier "multilingual-e5"
	

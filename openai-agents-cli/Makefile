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
	# uvx mypy src
	uv run mypy src

.PHONY: lint2
lint2:
	# Formatting
	uvx isort src
	uvx black src
	# Type Checking
	# uvx mypy src
	uv run mypy src
	# Linting
	uv run task flake8

###############################################################################
# Execution
# `PYTHONPATH=src` may be required to run the script
###############################################################################
.PHONY: run-example
run-example:
	uv run -m src.cli.main query-tech-guide --question "What is an advantage of using Python?"
	uv run -m src.cli.main query-tech-guide --chat --question "What is an advantage of using Rust?"
	uv run -m src.cli.main news-agent
	uv run -m src.cli.main embedding
	uv run -m src.cli.main search-similarity --id 1

# Ollama
.PHONY: run-with-ollama
run-with-ollama:
	# llama3.2
	uv run -m src.cli.main query-tech-guide --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --question "What is an advantage of using Rust?"
	uv run -m src.cli.main query-common --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --question "Give me the recipe for chicken curry"
	uv run -m src.cli.main search-similarity --tool ollama --model llama3.2 --id 1
	# Serarch web by Tavily
	uv run -m src.cli.main news-agent --tool ollama --model llama3.2
	# qwen3:8b
	uv run -m src.cli.main query-tech-guide --tool ollama --model qwen3:8b --embedding-model nomic-embed-text --chat --question "What is an advantage of using Rust?"
	# Prompting pattern
	uv run -m src.cli.main prompt-pattern --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --pattern zero-shot
	uv run -m src.cli.main prompt-pattern --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --pattern few-shot
	uv run -m src.cli.main prompt-pattern --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --pattern roll
	uv run -m src.cli.main prompt-pattern --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --pattern emotion
	uv run -m src.cli.main prompt-pattern --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --pattern cot
	uv run -m src.cli.main prompt-pattern --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --pattern cot2
	uv run -m src.cli.main prompt-pattern --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --pattern tot
	uv run -m src.cli.main prompt-pattern --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --pattern generated
	uv run -m src.cli.main prompt-pattern --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --pattern reflection
	uv run -m src.cli.main prompt-pattern --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --pattern meta
	uv run -m src.cli.main prompt-pattern --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --pattern prompt-chaining

# Note: run `lms load` first
.PHONY: run-with-lmstudio
run-with-lmstudio:
	uv run -m src.cli.main query-tech-guide --tool lmstudio --model llama3.2 --embedding-model text-embedding --chat --question "What is an advantage of using Rust?"

# WIP
.PHONY: run-vector
run-vector:
	# OpenAI
	# uv run -m src.cli.main query-common --question "What is an advantage of using Python?"
	# uv run -m src.cli.main query-common --question "What types of curry are there?"
	# uv run -m src.cli.main query-common --question "Give me the recipe for chicken curry"
	# uv run -m src.cli.main query-common --question "What is the largest river in Japan?"
	# uv run -m src.cli.main query-common --question "How far is it between Tokyo and Osaka?"
	# Ollama	
	uv run -m src.cli.main query-common --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --question "What is an advantage of using Python?"
	uv run -m src.cli.main query-common --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --question "What types of curry are there?"
	uv run -m src.cli.main query-common --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --question "Give me the recipe for chicken curry"
	uv run -m src.cli.main query-common --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --question "What is the largest river in Japan?"
	uv run -m src.cli.main query-common --tool ollama --model llama3.2 --embedding-model nomic-embed-text --chat --question "How far is it between Tokyo and Osaka?"

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
	lms load llama-3.2-3b-instruct --identifier "llama3.2"
	lms load text-embedding-nomic-embed-text-v1.5-embedding --identifier "text-embedding"
	
###############################################################################
# Ollama
###############################################################################

# ollama pull qwen3:30b-a3b
# ollama pull qwen3:8b

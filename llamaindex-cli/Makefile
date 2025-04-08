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
# .PHONY: run-example
# run-example:
# 	PYTHONPATH=src uv run -m src.cli.main docs-agent
# 	PYTHONPATH=src uv run -m src.cli.main docs-agent --storage text
#	PYTHONPATH=src uv run -m src.cli.main query-image-agent --image "./images/dog.jpg"
#	PYTHONPATH=src uv run -m src.cli.main calc-tool-agent --question "What is 20+(2*4)?"
#	PYTHONPATH=src uv run -m src.cli.main finance-tool-agent --company "NVIDIA"
#	PYTHONPATH=src uv run -m src.cli.main finance-tool-agent --company "NVIDIA" --tavily
#   # WIP:
#	PYTHONPATH=src uv run -m src.cli.main conversation-agent --question "Hello, how are you?"


# Use local LLM API server
# Note: run `lms load` first
.PHONY: run-with-local-llm
run-with-local-llm:
	PYTHONPATH=src uv run -m src.cli.main --local local-llm
	PYTHONPATH=src uv run -m src.cli.main --local docs-agent --model llama3
	PYTHONPATH=src uv run -m src.cli.main --local docs-agent --model llama3 --storage text
	PYTHONPATH=src uv run -m src.cli.main --local tech-question-agent --model llama3 --question "Rust"
	PYTHONPATH=src uv run -m src.cli.main --local tech-question-agent --model llama3 --question "Rust" --stream
	PYTHONPATH=src uv run -m src.cli.main --local tech-question-agent --model llama3 --question "Rust" --chat
	# WIP: find proper local LLM model
	#PYTHONPATH=src uv run -m src.cli.main --local calc-tool-agent --model hermes-3 --question "What is 20+(2*4)?"
	# WIP: find proper local LLM model
	#PYTHONPATH=src uv run -m src.cli.main --local query-image-agent --model llama3 --image "./images/dog.jpg"
	# WIP:
	#PYTHONPATH=src uv run -m src.cli.main --local conversation-agent --model llama3 --question "Hello, how are you?"

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
	# For embedding
	lms load text-embedding-nomic-embed-text-v1.5-embedding --identifier "text-embedding-ada-002"
	# For FunctionCallingLLM
	lms load hermes-2-pro-llama-3-8b --identifier "hermes-3"
	#lms load mistral-7b-instruct-v0.3 --identifier "mistral"

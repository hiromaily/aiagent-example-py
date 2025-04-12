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
# .PHONY: run-example
# run-example:
# 	PYTHONPATH=src uv run -m src.cli.main docs-agent
# 	PYTHONPATH=src uv run -m src.cli.main docs-agent --storage text
#	PYTHONPATH=src uv run -m src.cli.main query-image-agent --image "./images/dog.jpg"
#	PYTHONPATH=src uv run -m src.cli.main calc-tool-agent --question "What is 20+(2*4)?"
#	PYTHONPATH=src uv run -m src.cli.main finance-tool-agent --company "NVIDIA"
#	PYTHONPATH=src uv run -m src.cli.main finance-tool-agent --company "NVIDIA" --tavily
#	PYTHONPATH=src uv run -m src.cli.main conversation-agent


# Use local LLM API server
# Note: run `lms load` first
.PHONY: run-with-lmstudio
run-with-lmstudio:
	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev local-llm
	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev docs-agent --model llama3
	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev docs-agent --model llama3 --storage text
	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev tech-question-agent --model llama3 --question "Rust"
	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev tech-question-agent --model llama3 --question "Rust" --stream
	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev tech-question-agent --model llama3 --question "Rust" --chat
	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev conversation-agent --model llama3

	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev query-image-agent --model mistral --image "./images/dog.jpg"
	# LMStudio doesn't work: LLM must be a FunctionCallingLLM.
	#PYTHONPATH=src uv run -m src.cli.main --env=.env.dev calc-tool-agent --model mistral --question "What is 20+(2*4)?"
	#PYTHONPATH=src uv run -m src.cli.main --env=.env.dev finance-tool-agent --model mistral --company "NVIDIA" --tavily

.PHONY: run-with-ollama
run-with-ollama:
	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev tech-question-agent --model llama3.2 --question "Rust"
	# It doesn't work with image recognition
	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev query-image-agent --model llama3.2 --image "./images/dog.jpg"
	# FunctionCallingLLM
	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev calc-tool-agent --model llama3.2 --question "What is 20+(2*4)?"
	PYTHONPATH=src uv run -m src.cli.main --env=.env.dev finance-tool-agent --model llama3.2 --company "NVIDIA" --tavily


###############################################################################
# Utilities
###############################################################################
# remove `__pycache__` in each directory excluding `.venv`
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
	@# For embedding, model name needs to be manipulated as OpenAI specific models
	lms load text-embedding-nomic-embed-text-v1.5-embedding --identifier "text-embedding-ada-002"
	@# For image recognition
	lms load mistral-7b-instruct-v0.3 --identifier "mistral"
	@# For FunctionCallingLLM
	lms load functionary-7b-v2.1 --identifier "functionary"

# WIP
.PHONY: load-lms-experimental
load-lms-experimental:
	@# model name needs to be manipulated as OpenAI specific models
	lms load llama-3.2-3b-instruct --identifier "gpt-3.5-turbo"
	# PYTHONPATH=src uv run -m src.cli.main --env=.env.dev tech-question-agent --model gpt-3.5-turbo --question "Rust"
	lms load mistral-7b-instruct-v0.3 --identifier "gpt-4o-mini"
	# PYTHONPATH=src uv run -m src.cli.main --env=.env.dev calc-tool-agent --model gpt-4o-mini --question "What is 20+(2*4)?"

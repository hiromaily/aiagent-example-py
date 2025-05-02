# openai-agents-cli

A command-line interface for experimenting with local and remote LLM, Embeddings models through OpenAI client, and various prompting patterns.

## Overview

This project provides a flexible CLI interface for interacting with various LLM services and models, including:

- OpenAI API (Response API, Chat Completion API, Web Search API, etc.)
- Local LLM services (LM Studio, Ollama)
- Embedding models (text-embedding-ada-002, nomic-embed-text, etc.)
- Vector database integration (pgvector)

The architecture follows Clean Architecture principles with a clear separation between:

- Use cases
- Repository interfaces
- API clients
- CLI commands (Handler layer)

## Features

- ✅ Support for local and remote LLM models:
  - OpenAI API (`gpt-4o`, etc.)
  - LM Studio (local models via `baseUrl: "http://localhost:1234/v1"`)
  - Ollama (local models via `baseUrl: "http://localhost:11434/v1"`)
- ✅ Local embedding models
- ✅ Various prompting patterns for different use cases
- ✅ Vector database integration with pgvector
- ✅ Support Web Serach using Tavily / OpenAI Web Search API
- ✅ Multiple agent types:
  - Query agents
  - News agents
  - Prompting pattern agents

## Installation

### Prerequisites

- Python 3.13+
- uv package manager
- API keys for OpenAI (if using their services)
- Local setup for LM Studio or Ollama (if using local models)
- PostgreSQL with pgvector extension (if using vector database features)

### Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/hiromaily/aiagent-example-py
   cd openai-agents-cli
   uv pip install -e .
   ```

## How to run

See Makefile

```sh
uv run -m src.cli.main query-tech-guide --question "What is an advantage of using Python?"
```

## TODO

- [x] Local LLM mode
  - LM Studio: `baseUrl: "http://localhost:1234/v1"`
  - Ollama: `baseUrl: "http://localhost:11434/v1"`
- [x] Local embedding models
- [x] Various prompting patterns
- [ ] Various prompting patterns template + dynamic parameters
- Integrate Vector Database
  - [x] pgvector
- [ ] Cache for query

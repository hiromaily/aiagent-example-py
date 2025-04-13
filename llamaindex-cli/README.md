# llamaindex-cli

`llamaindex-cli` is a CLI tool designed to interact with LlamaIndex, a framework for building applications powered by LLMs. This CLI provides functionalities to integrate and manage various LLMs, tools, and APIs, enabling developers to build and test AI-driven workflows efficiently.

## Features

- **Local LLM Support**: Seamless integration with local LLMs like LM Studio and Ollama.
- **Tool Integration**: Supports adding tools like [Tavily](https://tavily.com/) for enhanced functionality.
- **Function Calling**: Enables function calling with compatible LLMs.
- **Customizable Models**: Allows specifying model configurations for OpenAI and other supported LLMs.

## Requirements

- [uv](https://github.com/astral-sh/uv) as Python package manager.
- OpenAI API Key
- LocalLLM tools like [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/)

## Getting Started

See [Makefile](./Makefile)

1. Clone the repository:

   ```sh
   git clone https://github.com/your-repo/llamaindex-cli.git
   cd llamaindex-cli
   ```

2. Install dependencies:

   ```sh
   uv venv .venv
   uv pip install -e .
   ```

3. Run the CLI:

   ```sh
   PYTHONPATH=src uv run -m src.cli.main --env=.env.dev <command> [options]
   ```

## Example Usage

To run a calculation tool agent with a specific model:

```sh
PYTHONPATH=src uv run -m src.cli.main --env=.env.dev calc-tool-agent --model gpt-4o-mini --question "What is 20+(2*4)?"
```

For more details, refer to the [LlamaIndex documentation](https://docs.llamaindex.ai/).

## Tools/Services used and LLM used for operation check

| Tool/API   | Chat      | embedding API                                  | Image Detection | Function Calling |
| ---------- | --------- | ---------------------------------------------- | --------------- | ---------------- |
| OpenAI API | gpt-4o    | text-embedding-ada-002                         | gpt-4o          | gpt-4o           |
| LM-Studio  | llama-3.2 | text-embedding-nomic-embed-text-v1.5-embedding | TBD             | not supported    |
| Ollama     | llama3.2  | nomic-embed-text                               | TBD             | llama3.2         |

## TODO

- [x] Local LLM mode
- [x] Integrate [Tavily](https://tavily.com/) to retrieve the ticker symbol.
  - [LlamaIndex: Adding other tools](https://docs.llamaindex.ai/en/stable/understanding/agent/tools/)
- [ ] Integrate Vector Database

## References

- [LlamaIndex: Agent Tools](https://llamahub.ai/?tab=tools)

### Example

- [python-ai-agents-demos](https://github.com/pamelafox/python-ai-agents-demos)
- [python-agents-tutorial](https://github.com/run-llama/python-agents-tutorial)

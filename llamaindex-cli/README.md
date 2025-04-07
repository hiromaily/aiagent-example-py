# llamaindex-cli

## TODO

- [x] Local LLM mode
  - LM Studio: `baseUrl: "http://localhost:1234/v1"`
  - [LlamaIndex: LM Studio](https://docs.llamaindex.ai/en/stable/examples/llm/lmstudio/)
  - Ollama: `baseUrl: "http://localhost:11434/v1"`
- [ ] Integrate Vector Database
- [ ] Integrate [Tavily](https://tavily.com/) to retrieve the ticker symbol.
  - [LlamaIndex: Adding other tools](https://docs.llamaindex.ai/en/stable/understanding/agent/tools/)

## [LlamaIndex: Agent Tools](https://llamahub.ai/?tab=tools)

## Issues

When using LM Studio with environment variable `OPENAI_API_KEY=lm-studio`, the following error occurs.
Why OpenAPI Key is required? `https://api.openai.com/v1/embeddings` seems to be called internally. And `https://api.openai.com/v1/chat/completions` as well.

```txt
AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: lm-studio. You can find
your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None,
'code': 'invalid_api_key'}}
```

[Fixed: Issued](https://github.com/run-llama/llama_index/issues/18349)

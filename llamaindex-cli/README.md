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

## Tips

model name which can be set into `OpenAI` is limited. Name is listed [here](https://github.com/run-llama/llama_index/blob/dac5aed708a4e5087cd54650ea8b7ae5bcc48a3a/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/utils.py#L73).

```py
from llama_index.llms.openai import OpenAI

OpenAI(model=model, api_key=api_key, temperature=temperature)
```

## Issues

### Wrong initialization for query

When using LM Studio with environment variable `OPENAI_API_KEY=lm-studio`, the following error occurs.
Why OpenAPI Key is required? `https://api.openai.com/v1/embeddings` seems to be called internally. And `https://api.openai.com/v1/chat/completions` as well.

```txt
AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: lm-studio. You can find
your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None,
'code': 'invalid_api_key'}}
```

[Fixed: Issued](https://github.com/run-llama/llama_index/issues/18349)

### WIP: `LLM must be a FunctionCallingLLM` error when running function calling.

First, LM Studio looks capable to use tools. [LM Studio: Tool Use](https://lmstudio.ai/docs/app/api/tools)

It would be occurred by `LMStudio` instance. `LMStudio` is not FunctionCallingLLM.

```python
# in `class OpenAI(FunctionCallingLLM)`
@property
def metadata(self) -> LLMMetadata:
    return LLMMetadata(
        context_window=openai_modelname_to_contextsize(self._get_model_name()),
        num_output=self.max_tokens or -1,
        is_chat_model=is_chat_model(model=self._get_model_name()),
        # this part must be important.
        is_function_calling_model=is_function_calling_model(
            model=self._get_model_name()
        ),
        model_name=self.model,
        # TODO: Temp for O1 beta
        system_role=MessageRole.USER
        if self.model in O1_MODELS
        else MessageRole.SYSTEM,
    )
```

These models would be allowed. [code](https://github.com/run-llama/llama_index/blob/dac5aed708a4e5087cd54650ea8b7ae5bcc48a3a/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/utils.py#L156)

```python
CHAT_MODELS = {
    **O1_MODELS,
    **GPT4_MODELS,
    **TURBO_MODELS,
    **AZURE_TURBO_MODELS,
```

## Example

- [python-ai-agents-demos](https://github.com/pamelafox/python-ai-agents-demos)

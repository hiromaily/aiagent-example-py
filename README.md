# aiagent-example-py

Various examples of AI Agents.

- [LlamaIndex](https://www.llamaindex.ai/)
   LlamaIndex (GPT Index) is a data framework for your LLM application. Building with LlamaIndex typically involves working with LlamaIndex core and a chosen set of integrations (or plugins)
- [PydanticAI](https://ai.pydantic.dev/)
   PydanticAI is a Python agent framework designed to make it less painful to build production grade applications with Generative AI.
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
   The OpenAI Agents SDK enables you to build agentic AI apps in a lightweight, easy-to-use package with very few abstractions. It's a production-ready upgrade of our previous experimentation for agents, Swarm.
- [Rig](https://rig.rs/)
   Rig is a Rust library for building scalable, modular, and ergonomic LLM-powered applications.

## Planning

- [CrewAI](https://docs.crewai.com/introduction)
   CrewAI is a lean, lightning-fast Python framework built entirely from scratch—completely independent of LangChain or other agent frameworks.
- [DSPy](https://dspy.ai/)
   DSPy is the framework for programming—rather than prompting—language models. It allows you to iterate fast on building modular AI systems and offers algorithms for optimizing their prompts and weights, whether you're building simple classifiers, sophisticated RAG pipelines, or Agent loops.
- [Mastra](https://github.com/mastra-ai/mastra)
  - Planning
- [LangChain](https://www.langchain.com/)
  - [Pending] For a while, keep watching on stability of progress because of [Article: Why Developers are Quitting LangChain](https://analyticsindiamag.com/ai-features/why-developers-are-quitting-langchain/)

## Tools/Services used and LLM used for operation check

| Tool/API   | Chat      | embedding API                                  | Image Detection | Function Calling |
| ---------- | --------- | ---------------------------------------------- | --------------- | ---------------- |
| OpenAI API | gpt-4o    | text-embedding-ada-002                         | gpt-4o          | gpt-4o           |
| LM-Studio  | llama-3.2 | text-embedding-nomic-embed-text-v1.5-embedding | TBD             | not supported    |
| Ollama     | llama3.2  | nomic-embed-text                               | TBD             | llama3.2         |

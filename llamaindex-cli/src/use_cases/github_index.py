"""Github Index Use Case."""

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding

# from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms import LLM
from llama_index.core.vector_stores import SimpleVectorStore
from loguru import logger

from infrastructure.storages.github import GithubDocumentList


class GithubIndex:
    """Github Index Use Case."""

    def __init__(self, llm: LLM, embed_model: BaseEmbedding, github_docs: GithubDocumentList) -> None:
        """Initialize the Github Index with a LLM."""
        self._llm = llm
        self._embed_model = embed_model
        self._github_docs = github_docs

    def index(self, output_dir: str = "storage") -> None:
        """Ask the question by chat()."""
        documents = self._github_docs.get_document()

        logger.debug("create index")
        # Note: `ServiceContext`` is deprecated
        # service_context = ServiceContext.from_defaults(embed_model=self._embed_model)

        # Note: VectorStoreIndex.from_documents() returns an index
        VectorStoreIndex.from_documents(
            documents,
            embed_model=self._embed_model,
        )

        logger.debug("store index")
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.persist(persist_dir=output_dir)
        logger.debug(f"index saved in {output_dir}")

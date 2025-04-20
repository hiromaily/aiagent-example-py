"""Github Index Use Case."""

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms import LLM
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from loguru import logger

from infrastructure.documents.github import GithubDocumentList


class GithubIndex:
    """Github Index Use Case."""

    # Github Repo Reader
    # https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/

    def __init__(
        self,
        llm: LLM,
        embed_model: BaseEmbedding,
        github_docs: GithubDocumentList,
        vector_store: BasePydanticVectorStore,
    ) -> None:
        """Initialize the Github Index with a LLM."""
        self._llm = llm
        self._embed_model = embed_model
        self._github_docs = github_docs
        self._vector_store = vector_store
        if isinstance(vector_store, SimpleVectorStore):
            self._output_dir = "storage/github/docs"
        else:
            self._output_dir = None

    def store_index(self) -> None:
        """Create github document and vector index and store it."""
        logger.debug("start calling GithubDocumentList")
        # 1. get documents
        documents = self._github_docs.get_document()
        logger.debug("end calling GithubDocumentList")

        logger.debug("create index")
        # Note: `ServiceContext`` is deprecated
        # create storage context with specific storage
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        # create vector index with `embed_model`
        index = VectorStoreIndex.from_documents(
            documents, embed_model=self._embed_model, storage_context=storage_context
        )

        logger.debug("store index")
        if self._output_dir is not None:
            index.storage_context.persist(persist_dir=self._output_dir)
        else:
            index.storage_context.persist()
        logger.debug("index saved")

    def _load_saved_index(self) -> BaseIndex:
        """Load saved index."""
        # https://docs.llamaindex.ai/en/stable/module_guides/storing/save_load/
        if isinstance(self._vector_store, SimpleVectorStore):
            storage_context = StorageContext.from_defaults(persist_dir=self._output_dir)
            return load_index_from_storage(storage_context, embed_model=self._embed_model)
        return VectorStoreIndex.from_vector_store(self._vector_store, embed_model=self._embed_model)

    def search_index(self, question: str) -> None:
        """Ask the question from github docs."""
        logger.debug("load index from storage")
        index = self._load_saved_index()
        query = index.as_query_engine(self._llm)

        logger.debug(f"query: {question}")
        response = query.query(question)
        print(response)

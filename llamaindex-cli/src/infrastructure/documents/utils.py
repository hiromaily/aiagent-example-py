"""Storage utils."""

from typing import cast

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document


def files_to_node(docs: list[Document], chunk_size: int = 512) -> list[BaseNode]:
    """Convert github docs document to node."""
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
    node = node_parser.get_nodes_from_documents(docs, show_progress=False)
    return cast("list[BaseNode]", node)

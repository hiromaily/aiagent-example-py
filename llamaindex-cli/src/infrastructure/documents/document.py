"""Create text storage."""

from enum import Enum
from typing import cast

from llama_index.core import Document, SimpleDirectoryReader


class StorageMode(Enum):
    """Storage mode enum."""

    TEXT = "text"
    DIR = "dir"

    @classmethod
    def from_str(cls, mode_str: str) -> "StorageMode":
        """Change string to StorageMode."""
        for mode in cls:
            if mode.value == mode_str:
                return mode
        msg = f"'{mode_str}' is not a valid StorageMode"
        raise ValueError(msg)


class DocumentList:
    """Document list class."""

    def __init__(self, mode: StorageMode) -> None:
        """Initialize the DocumentList with the mode."""
        self.mode = mode

    def get_document(self) -> list[Document]:
        """Get the document based on the mode."""
        if self.mode == StorageMode.TEXT:
            return [
                Document(text="LlamaIndex is a data framework for LLM applications."),
                Document(text="It provides tools for indexing, querying, and augmenting LLM capabilities."),
                Document(text="LlamaIndex supports various data connectors and index types."),
            ]
        if self.mode == StorageMode.DIR:
            return cast("list[Document]", SimpleDirectoryReader("storage/news").load_data())

        msg = f"'{self.mode}' is not a valid StorageMode"
        raise ValueError(msg)


# def get_const_docs() -> list[Document]:
#     return [
#         Document(text="LlamaIndex is a data framework for LLM applications."),
#         Document(text="It provides tools for indexing, querying, and augmenting LLM capabilities."),
#         Document(text="LlamaIndex supports various data connectors and index types."),
#     ]

# def get_dir_docs() -> list[Document]:
#     return SimpleDirectoryReader("storage/news").load_data()

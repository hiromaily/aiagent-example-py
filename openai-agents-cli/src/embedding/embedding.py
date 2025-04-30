"""Embedding for debug Use Case."""

import json
from pathlib import Path

from entities.embedding import Embedding


def load_embedding(file_name: str) -> list[Embedding]:
    """Loading embedding JSON file."""
    # load JSON file
    # with open("storage/embedding01.json") as f:
    with Path(file_name).open("r") as f:
        embeddings_data = json.load(f)

    return [Embedding(embedding=embeddings_data, index=0, object_type="embedding")]
    # debug
    # for embedding in embeddings_list:
    #     print(
    #         f"Embedding {embedding.index}: {embedding.embedding[:10]}... "
    #         f"(total {len(embedding.embedding)} values), {embedding.object}"
    #     )

"""Create Github storage."""

from typing import cast

from llama_index.core import Document
from llama_index.readers.github import GithubRepositoryReader


class GithubDocumentList:
    """Github Document list class."""

    def __init__(self, token: str, owner: str, repo: str) -> None:
        """Initialize the DocumentList with the mode."""
        self._github_reader = self._build_github_reader(token, owner, repo)

    def _build_github_reader(self, token: str, owner: str, repo: str) -> list[Document]:
        reader = GithubRepositoryReader(
            github_client=token,
            owner=owner,
            repo=repo,
            # filter_directories=(["docs"], GithubRepositoryReader.FilterType.INCLUDE),
            filter_file_extensions=([".md"], GithubRepositoryReader.FilterType.INCLUDE),
        ).load_data(branch="main")
        return cast("list[Document]", reader)

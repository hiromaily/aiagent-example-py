"""Create Github storage."""

from typing import cast

from llama_index.core import Document
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from llama_index.readers.github.repository.github_client import (
    BaseGithubClient,
)


class GithubDocumentList:
    """Github Document list class."""

    def __init__(self, token: str, owner: str, repo: str) -> None:
        """Initialize the DocumentList with the mode."""
        github_client = GithubClient(github_token=token, verbose=True)
        self._github_reader = self._build_github_reader(github_client, owner, repo)

    def _build_github_reader(self, github_client: BaseGithubClient, owner: str, repo: str) -> list[Document]:
        reader = GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            # filter_directories=(["docs"], GithubRepositoryReader.FilterType.INCLUDE),
            filter_file_extensions=([".md"], GithubRepositoryReader.FilterType.INCLUDE),
            verbose=True,
        ).load_data(branch="main")
        return cast("list[Document]", reader)

    def get_document(self) -> list[Document]:
        """Get the github docs document."""
        return self._github_reader

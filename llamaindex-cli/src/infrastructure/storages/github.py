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
        self._github_client = GithubClient(github_token=token, verbose=True)
        self._owner = owner
        self._repo = repo
        self._github_documents: None | list[Document] = None

    # very high cost if repository is large
    def build_github_documents(self, github_client: BaseGithubClient, owner: str, repo: str) -> list[Document]:
        """Build the Github documents."""
        reader = GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            filter_directories=(
                ["programming/rust"],
                GithubRepositoryReader.FilterType.INCLUDE,
            ),  # enabled when testing
            filter_file_extensions=([".md"], GithubRepositoryReader.FilterType.INCLUDE),
            verbose=True,
            timeout=180,
        ).load_data(branch="main")
        return cast("list[Document]", reader)

    def get_document(self) -> list[Document]:
        """Get the github docs document."""
        if self._github_documents is None:
            self._github_documents = self.build_github_documents(self._github_client, self._owner, self._repo)
        return self._github_documents

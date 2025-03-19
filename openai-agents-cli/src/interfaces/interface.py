"""This module defines the base interface for the application."""

from abc import ABC, abstractmethod


class BaseInterface(ABC):
    """Base Interface Class."""

    @abstractmethod
    def some_operation(self) -> None:
        """Some operation."""

    @abstractmethod
    def another_operation(self, param: str) -> str:
        """Another operation."""


# Additional interfaces can be defined here as needed.

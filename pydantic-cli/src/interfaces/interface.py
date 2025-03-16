from abc import ABC, abstractmethod


class BaseInterface(ABC):
    @abstractmethod
    def some_operation(self) -> None:
        pass

    @abstractmethod
    def another_operation(self, param: str) -> str:
        pass


# Additional interfaces can be defined here as needed.

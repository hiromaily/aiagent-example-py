from typing import Any


class UseCase:
    def __init__(self) -> None:
        pass

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses should implement this method.")


class ExampleUseCase(UseCase):
    def execute(self, input_data: str) -> str:
        # Implement the specific logic for this use case
        return f"Processed: {input_data}"

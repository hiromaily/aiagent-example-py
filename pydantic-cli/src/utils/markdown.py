"""This is utility modules."""

from pydantic import BaseModel


def to_markdown(data: str, indent: int = 0) -> str:
    """Convert a dictionary, list or pydantic model to a markdown string. The function is recursive.

    :param data: The data to convert to markdown.
    :param indent: The indentation level.
    :return: The markdown string.
    """
    markdown = ""
    if isinstance(data, BaseModel):
        data = data.model_dump()
    if isinstance(data, dict):
        for key, value in data.items():
            markdown += f"{'#' * (indent + 2)} {key.upper()}\n"
            if isinstance(value, (dict, list, BaseModel)):
                markdown += to_markdown(value, indent + 1)
            else:
                markdown += f"{value}\n\n"
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list, BaseModel)):
                markdown += to_markdown(item, indent)
            else:
                markdown += f"- {item}\n"
        markdown += "\n"
    else:
        markdown += f"{data}\n\n"
    return markdown

"""Transient list/text modes for scalar suggestion fields."""

from dataclasses import replace

from .description import FormDescription, FormFieldDescription
from .model import FormWritePath


def present(
    description: FormDescription, text_fields: set[FormWritePath]
) -> FormDescription:
    """Project display modes without modifying the input or option declarations."""

    def field(item: FormFieldDescription) -> FormFieldDescription:
        if not any(command.name == "toggle_text" for command in item.commands):
            return item
        text = item.path in text_fields
        return replace(
            item,
            selector=not text,
            commands=tuple(
                replace(command, label="Choose from list" if text else "Edit as text")
                if command.name == "toggle_text"
                else command
                for command in item.commands
            ),
        )

    return replace(
        description,
        target=field(description.target) if description.target is not None else None,
        children=tuple(field(item) for item in description.children),
    )

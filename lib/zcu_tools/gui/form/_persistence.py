"""The single JSON wire format for raw input and explicit selections."""

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from .model import FormWritePath


class _Selection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: list[str | int]
    value: JsonValue


class _Draft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    data: dict[str, JsonValue]
    selections: list[_Selection]


def encode(
    data: dict[str, JsonValue], selections: dict[FormWritePath, JsonValue]
) -> str:
    """Encode detached raw data and typed choices without inspecting annotations."""
    return _Draft(
        data=data,
        selections=[
            _Selection(path=list(path), value=value)
            for path, value in selections.items()
        ],
    ).model_dump_json()


def decode(text: str) -> tuple[dict[str, JsonValue], dict[FormWritePath, JsonValue]]:
    """Decode exactly {data,selections}; malformed or duplicate paths raise ValueError.

    Annotation compatibility belongs to restoration. No declaration, sources,
    options, converter or value validation participates in JSON decoding.
    """
    try:
        draft = _Draft.model_validate_json(text)
    except ValidationError as exc:
        raise ValueError(f"Invalid FormInput JSON: {exc}") from exc
    selections: dict[FormWritePath, JsonValue] = {}
    for choice in draft.selections:
        path = tuple(choice.path)
        if not path or path in selections:
            raise ValueError("Invalid or duplicate saved selection path")
        selections[path] = choice.value
    return draft.data, selections

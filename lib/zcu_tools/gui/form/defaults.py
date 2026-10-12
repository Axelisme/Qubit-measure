"""Public Pydantic static defaults for input authors and branch creation."""

from pydantic import BaseModel, JsonValue

from ._json import default_json


def static_defaults(model: type[BaseModel]) -> dict[str, JsonValue]:
    """Encode declared defaults as fresh input JSON without validating a model.

    ``model`` may be any Pydantic BaseModel class, including persistent Specs.
    Required fields and defaults equal to None remain absent so before validators
    can derive omitted values. Default factories run once per call. Values use
    the form complex envelope and detached JSON containers. Invalid non-finite
    or non-JSON defaults raise ValueError; factory failures propagate. No source
    lookup, evaluation or I/O is performed by this function.
    """
    result: dict[str, JsonValue] = {}
    for name, info in model.model_fields.items():
        if not info.is_required():
            value = info.get_default(call_default_factory=True)
            if value is not None:
                result[name] = default_json(value)
    return result

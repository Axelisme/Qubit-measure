"""Pure scalar input rules shared by input editing and binding resolution."""

from .model import DirectValue, ScalarSpec, require_finite_scalar


def parse_scalar_text(spec: ScalarSpec, text: str) -> DirectValue:
    if spec.type not in (int, float, complex, str):
        raise TypeError(f"Text input is unsupported for {spec.type.__name__}")
    if not text.strip() and (spec.optional or spec.type is not str):
        return DirectValue(None, raw=text)
    try:
        parsed = spec.type(text.strip() if spec.optional else text)
        if isinstance(parsed, (float, complex)):
            require_finite_scalar(parsed)
    except ValueError as exc:
        return DirectValue(None, raw=text, error=str(exc))
    return DirectValue(parsed, raw=text)


def validate_direct_scalar(spec: ScalarSpec, value: DirectValue) -> None:
    raw = value.value
    if raw is None:
        return
    if type(raw) is not spec.type:
        raise TypeError(
            f"ScalarField {spec.label!r} expects "
            f"{spec.type.__name__}, got {type(raw).__name__}"
        )
    if isinstance(raw, (float, complex)):
        require_finite_scalar(raw)


def coerce_scalar_result(
    value: int | float | complex, type_: type
) -> int | float | complex:
    if isinstance(value, bool):
        raise RuntimeError("Expression evaluator returned bool instead of a number")
    if type_ is complex:
        result = complex(value)
        require_finite_scalar(result)
        return result
    if isinstance(value, complex):
        raise RuntimeError("Complex expression result cannot target a real field")
    if type_ is float:
        result = float(value)
        require_finite_scalar(result)
        return result
    if type_ is int:
        if not float(value).is_integer():
            raise RuntimeError(f"Expression result {value!r} is not an integer")
        return int(value)
    raise RuntimeError(f"Eval mode only supports int, float or complex, got {type_!r}")

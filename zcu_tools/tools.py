from collections.abc import MutableMapping
from typing import Literal, Optional


def deepupdate(
    d: dict, u: dict, behavior: Literal["error", "force", "ignore"] = "error"
):
    def conflict_handler(d, u, k):
        if behavior == "error":
            raise KeyError(f"Key {k} already exists in {d}.")
        elif behavior == "force":
            d[k] = u[k]
        elif behavior == "ignore":
            pass
        else:
            raise ValueError(f"Unknown behavior: {behavior}")

    assert isinstance(d, MutableMapping), f"d should be dict, got {d}"
    assert isinstance(u, MutableMapping), f"u should be dict, got {u}"
    for k, v in u.items():
        if k not in d:
            d[k] = v
        elif isinstance(v, MutableMapping) and isinstance(d[k], MutableMapping):
            deepupdate(d[k], v, behavior)
        else:
            conflict_handler(d, u, k)


# type cast all numpy types to python types
def numpy2number(obj):
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {k: numpy2number(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy2number(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def make_sweep(
    start: float,
    stop: Optional[float] = None,
    expts: Optional[int] = None,
    step: Optional[float] = None,
    force_int: bool = False,
) -> dict:
    assert (
        stop is not None or step is not None or expts is not None
    ), "Not enough information to define a sweep."

    error_msg = "Not enough information to define a sweep."
    if expts is None:
        assert step is not None, error_msg
        expts = int(stop - start) // step
    elif step is None:
        assert expts is not None, error_msg
        step = (stop - start) / expts

    if force_int:
        start = int(start)
        step = int(step)
        expts = int(expts)

    stop = start + step * expts

    assert expts > 0, f"expts must be greater than 0, but got {expts}"
    assert step != 0, f"step must not be zero, but got {step}"

    return {"start": start, "stop": stop, "expts": expts, "step": step}

from collections.abc import MutableMapping
from typing import Optional


def deepupdate(d: dict, u: dict, overwrite: bool = False):
    for k, v in u.items():
        if isinstance(v, MutableMapping):
            d.setdefault(k, {})
            deepupdate(d[k], v, overwrite=overwrite)
        elif k not in d or overwrite:
            d[k] = v
        else:
            raise KeyError(f"Key {k} already exists in {d}.")


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

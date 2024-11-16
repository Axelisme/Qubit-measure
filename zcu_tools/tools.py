from collections.abc import MutableMapping


def deepupdate(d: dict, u: dict):
    for k, v in u.items():
        if isinstance(v, MutableMapping):
            d.setdefault(k, {})
            deepupdate(d[k], v)
        else:
            d[k] = v


def make_sweep(
    start: float,
    stop: float | None = None,
    expts: int | None = None,
    step: float | None = None,
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

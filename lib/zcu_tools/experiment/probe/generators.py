"""Caller-model synthesis and population-based raw IQ sampling."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


def synthesize(
    model: Callable[..., NDArray[np.number]],
    xs: NDArray[np.number],
    params: Sequence[float],
    *,
    noise_std: float,
    seed: int,
) -> NDArray[np.number]:
    """Call model(xs, *params) and add independent Gaussian noise per element.

    xs supplies the model's coordinates; params follows that callable's native
    signature and units. noise_std is the finite, nonnegative standard deviation
    in output units. Complex output gets independent noise of that standard
    deviation in each real and imaginary component. seed initializes a local
    NumPy RNG without changing global random state.

    The model determines the output shape, not xs. NumPy addition determines
    the output dtype. Only finite, nonempty xs and noise_std are validated;
    violations raise ValueError. Model parameters and output are not validated,
    converted or repaired. Model and NumPy exceptions propagate unchanged.
    Synthesis does not modify inputs; the supplied model owns its side effects.

    Example using the fitting owner's parameter order:
        >>> from zcu_tools.analysis.fitting import expfunc
        >>> xs = np.linspace(0.0, 10.0, 101)
        >>> ys = synthesize(expfunc, xs, [0.2, 0.8, 4.0], noise_std=0.01, seed=17)
    """
    if xs.size == 0 or not np.all(np.isfinite(xs)):
        raise ValueError("xs must be finite and nonempty")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    values = model(xs, *params)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, values.shape)
    if np.iscomplexobj(values):
        return values + noise + 1j * rng.normal(0.0, noise_std, values.shape)
    return values + noise

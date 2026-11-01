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


def sample_iq_shots(
    excited_population: NDArray[np.float64],
    *,
    g_center: complex,
    e_center: complex,
    width: float,
    shots: int,
    seed: int,
) -> NDArray[np.complex128]:
    """Draw raw complex128 IQ with shape (len(excited_population), shots).

    excited_population is a finite 1D array of probabilities in [0, 1], one per
    row; it may be empty. Each shot independently selects e_center with that
    row's probability, otherwise g_center. Both centers must be finite complex
    IQ coordinates. width is the finite, nonnegative Gaussian standard
    deviation in each quadrature, in the centers' IQ units. Independent
    N(0, width**2) noise is added to I and Q, without a grid or truncation.
    shots is a nonnegative integer, not bool; zero returns zero columns.

    seed initializes a local NumPy RNG without changing global random state.
    Invalid population, centers, width or shots raise ValueError. NumPy seed
    and sampling errors propagate unchanged. No inputs are modified.
    Use populations [0, 1] for GE, [p, 1-p] for symmetric preparation errors,
    or a population curve for a sweep.

    Example sampling pure ground and excited preparations:
        >>> iq = sample_iq_shots(np.array([0.0, 1.0]), g_center=-1+0j,
        ...                      e_center=1+0j, width=0.2, shots=1000, seed=17)
        >>> iq.shape
        (2, 1000)
    """
    if (
        excited_population.ndim != 1
        or not np.all(np.isfinite(excited_population))
        or np.any((excited_population < 0) | (excited_population > 1))
    ):
        raise ValueError("excited_population must be finite, 1D and in [0, 1]")
    if not np.isfinite(g_center) or not np.isfinite(e_center):
        raise ValueError("g_center and e_center must be finite")
    if not np.isfinite(width) or width < 0:
        raise ValueError("width must be finite and nonnegative")
    if (
        isinstance(shots, bool)
        or not np.issubdtype(type(shots), np.integer)
        or shots < 0
    ):
        raise ValueError("shots must be a nonnegative integer, not bool")
    rng = np.random.default_rng(seed)
    excited = rng.random((len(excited_population), shots)) < excited_population[:, None]
    centers = np.where(excited, np.complex128(e_center), np.complex128(g_center))
    return (
        centers
        + rng.normal(0.0, width, centers.shape)
        + 1j * rng.normal(0.0, width, centers.shape)
    )

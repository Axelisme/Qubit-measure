"""Shared real-acquire helpers for the measurement Nodes (RB-2).

Every measurement Node's ``produce`` follows the same lower-layer recipe
(``experiment/v2/autofluxdep`` ``measure_fn`` + ``run``): write this flux point's
value into ``cfg.dev`` BY NAME → ``setup_devices`` to push it → build a
``ModularProgramV2`` module list with the swept axis → ``.acquire`` with a
running-average ``round_hook`` + cooperative-stop / SNR-early-stop
``stop_checkers`` → collapse the raw IQ to a 1-D complex trace → rotate to real.

These pieces are identical across qubit_freq / lenrabi / ro_optimize / t1 / t2* /
mist, so they live here once rather than being copy-pasted into seven Nodes. The
Node-specific parts (which modules, which sweep, which fitter, the Patch) stay in
each Node.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment.v2.utils import snr_checker
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.utils.process import rotate2real

# How many running-average rounds a freshly GUI-placed Node acquires by default.
# A real acquire averages many shots; ``rounds`` is the round count the cfg passes
# to ``.acquire`` (the running average settles round by round). The controller
# seeds this onto a GUI-placed Node; a directly-constructed Node (tests) overrides
# it via params. Kept here (the real-acquire home) rather than scattered in cfg.
DEFAULT_ROUNDS = 10


def parse_linear_axis(
    spec: Any, default: tuple[float, float, int]
) -> NDArray[np.float64]:
    """Parse a free-text "start,stop,npts" sweep axis (or a 3-tuple) to a linspace.

    The prototype's sweep fields are free text; a malformed value degrades to
    ``default`` rather than failing the sweep (shared by t1 / lenrabi / t2ramsey /
    t2echo / mist, whose trailing axis is a simple linspace)."""
    try:
        if isinstance(spec, str) and spec.strip():
            start, stop, npts = spec.split(",")
            lo, hi, n = float(start), float(stop), int(npts)
        elif isinstance(spec, (tuple, list)) and len(spec) == 3:
            lo, hi, n = float(spec[0]), float(spec[1]), int(spec[2])
        else:
            lo, hi, n = default
    except (ValueError, TypeError):
        lo, hi, n = default
    return np.linspace(lo, hi, max(2, n))


def is_good_fit(
    real: NDArray[np.float64], fit_curve: NDArray[np.float64], threshold: float = 0.2
) -> bool:
    """Whether a fit is good enough to trust (the runner module's mean_err gate).

    Compares the mean absolute residual to the fit's peak-to-peak span: a good fit
    tracks the signal so the residual is a small fraction of the span. At an
    SNR-trough flux point the acquired signal is mostly noise — the fitted curve is
    nearly flat (tiny span) while the residual is large, so this returns False, and
    the Node omits that point's provides key (no downstream contamination) and skips
    calibration. Mirrors ``mean_err < threshold * ptp(fit)`` per experiment.
    """
    fit = np.asarray(fit_curve, dtype=np.float64)
    span = float(np.ptp(fit))
    if span <= 0 or not np.all(np.isfinite(fit)):
        return False
    residual = float(np.mean(np.abs(np.asarray(real, dtype=np.float64) - fit)))
    return residual < threshold * span


def axis_to_sweep(axis: NDArray[np.float64]) -> SweepCfg:
    """Reconstruct the ``SweepCfg`` a swept trailing axis was sampled from.

    A Result stores its trailing axis as an explicit linspace; the program-side
    sweep is a ``SweepCfg`` (start/stop/expts/step). Rebuilds it from the axis
    endpoints + length so the FPGA sweep matches the Result's columns exactly
    (shared by every 1-D node: length / relax-time / delay-time / gain axes)."""
    arr = np.asarray(axis, dtype=np.float64)
    expts = int(arr.shape[0])
    start = float(arr[0])
    stop = float(arr[-1])
    step = 0.0 if expts == 1 else (stop - start) / (expts - 1)
    return SweepCfg(start=start, stop=stop, expts=expts, step=step)


def set_flux_by_name(
    cfg_dev: MutableMapping[str, Any] | None, name: str, value: float
) -> None:
    """Write ``value`` into ``cfg.dev[name]`` (the picked flux device, by NAME).

    ``cfg.dev`` is keyed by device name (``GlobalDeviceManager.get_all_info``); the
    GUI flux picker stores a device *name* (e.g. the auto-provisioned ``fake_flux``).
    The lower layer's ``set_flux_in_dev_cfg`` resolves by ``label`` (``flux_dev``),
    a different dimension the GUI's picked device need not carry — writing by name
    is the dimension that cannot silently miss. Fast Fail if cfg.dev is empty or the
    named device is absent (not connected / not in the device manager), so a flux
    that never reaches the device surfaces as an error rather than a constant fit.
    """
    if not cfg_dev or name not in cfg_dev:
        known = sorted(cfg_dev) if cfg_dev else []
        raise KeyError(
            f"flux device {name!r} not found in cfg.dev "
            f"(known: {known}); is it connected?"
        )
    cfg_dev[name] = cfg_dev[name].with_updates(value=value)


def require_flux_device(env: RunEnv, exp_name: str) -> str:
    """Return the picked flux device name, Fast Fail if none is set.

    A real flux sweep must drive a device; an unset ``flux_device`` means the
    user never picked a flux source, so the sweep cannot push this point's value.
    """
    if env.flux_device is None:
        raise RuntimeError(
            f"{exp_name} real acquire needs a flux device picked "
            "(state.flux_device_name); none is set"
        )
    return env.flux_device


def acquire_to_complex(raw: Any) -> NDArray[np.complex128]:
    """Collapse a ``.acquire`` round/result into a 1-D complex trace.

    ``.acquire`` (and its ``round_hook`` running average) returns ``list[NDArray]``
    per readout; the first readout's first row is the (n, 2) I/Q pair vs the swept
    axis. Mirrors the runner's ``default_raw2signal_fn`` (``raw[0][0].dot([1,
    1j])``)."""
    return np.asarray(raw[0][0], dtype=np.float64).dot([1, 1j])


def signal2real_flip(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """PCA-rotate to the real axis, normalise to [0, 1], orient as a decay.

    The lower layer's ``t1_signal2real`` / ``lenrabi_signal2real`` /
    ``t2*_signal2real`` share one shape: rotate, min-max normalise, then flip so
    the trace starts high (``init_val`` above the midpoint). The decay / rabi /
    fringe fitters are baseline-agnostic, but flipping keeps the curve in the
    orientation the lower layer fits, so the fitted scalar matches."""
    real = rotate2real(signals.astype(np.complex128)).real
    lo, hi = float(real.min()), float(real.max())
    real = (real - lo) / (hi - lo + 1e-12)
    if real[0] < 0.5:  # init below midpoint → flip so the decay starts high
        real = 1.0 - real
    return real


class SnrProbe:
    """A minimal ``ctx``-shaped shim so the lower-layer ``snr_checker`` can read
    the running acquire trace. ``snr_checker`` only touches ``ctx.value`` — the
    Node updates ``value`` each round so the SNR early-stop sees the latest average.
    """

    value: NDArray[np.complex128] | None = None


def earlystop_snr(params: Any) -> float | None:
    """The SNR early-stop threshold from a Node's params, or None (no early-stop).

    Mirrors the lower-layer task's ``earlystop_snr``. The prototype's param field
    is free text, so an unset / unparseable value disables early-stop rather than
    failing the acquire."""
    try:
        value = params.get("earlystop_snr")
    except AttributeError:
        return None
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def build_stop_checkers(
    env: RunEnv,
    probe: SnrProbe,
    signal2real_fn: Callable[[np.ndarray], np.ndarray],
) -> list[Callable[[], bool]]:
    """The ``stop_checkers`` list every measurement acquire passes to ``.acquire``.

    Threads the run's cooperative cancel (``env.should_stop``) + the SNR early-stop
    (``snr_checker`` reading the running average via ``probe``) — exactly the lower
    layer's ``stop_checkers=[ctx.is_stop, snr_checker(ctx, ...)]``."""
    checkers: list[Callable[[], bool]] = []
    if env.should_stop is not None:
        checkers.append(env.should_stop)
    checkers.append(snr_checker(probe, earlystop_snr(env.params), signal2real_fn))
    return checkers

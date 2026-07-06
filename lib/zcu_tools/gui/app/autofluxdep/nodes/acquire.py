"""Shared real-acquire helpers for the measurement Nodes (RB-2).

Every measurement Node's ``produce`` follows the same lower-layer recipe
(``experiment/v2/autofluxdep`` Schedule acquire + ``run``): write this flux point's
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

import logging
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from zcu_tools.experiment.v2.runner import ProgramBuilder, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import estimate_snr, snr_checker
from zcu_tools.gui.app.autofluxdep.cfg import IntSpec
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.defaults import (
    GenerationField,
    logical_generation_field,
)
from zcu_tools.gui.app.autofluxdep.profiling import PerfStats, elapsed_ms, perf_now
from zcu_tools.program.v2 import ModularProgramV2, SweepCfg
from zcu_tools.utils.process import rotate2real

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.cfg import NodeCfgSchema

logger = logging.getLogger(__name__)
_ROUND_HOOK_PERF = PerfStats("worker.round_hook", logger, slow_ms=20.0)
_DECAY_SCALAR_RESIDUAL_RATIO = 0.1
_DECAY_SCALAR_MAX_SWEEP_FACTOR = 2.0
DEFAULT_ACQUIRE_RETRY = 3


@dataclass(frozen=True)
class ScheduleAcquireResult:
    """Node-facing result of a Schedule-backed program acquire."""

    signal: NDArray[Any] | None
    stopped: bool = False


def acquire_retry_generation_field(
    *, group: str = "acquisition", group_label: str | None = None
) -> GenerationField:
    """Return the common node-level acquire retry generation knob."""
    return logical_generation_field(
        "acquire_retry",
        IntSpec(label="retry"),
        DEFAULT_ACQUIRE_RETRY,
        group=group,
        group_label=group_label,
    )


def acquire_retry(env: RunEnv) -> int:
    """Return this node's acquire retry budget, Fast Failing invalid values."""
    value = env.knobs().get("acquire_retry", DEFAULT_ACQUIRE_RETRY)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(
            f"{env.node_name or 'node'} acquire_retry must be an integer, got {value!r}"
        )
    if value < 0:
        raise RuntimeError(
            f"{env.node_name or 'node'} acquire_retry must be non-negative, got {value}"
        )
    return value


def run_schedule_acquire(
    *,
    env: RunEnv,
    cfg: object,
    signal_shape: int | tuple[int, ...],
    dtype: DTypeLike,
    configure_builder: Callable[[ProgramBuilder[Any]], object],
    raw2signal_fn: Callable[[Any], NDArray[Any]],
    on_update: Callable[[NDArray[Any]], None] | None = None,
    program_cls: Callable[..., Any] = ModularProgramV2,
    stop_checkers: list[Callable[[], bool]] | None = None,
    progress_label: str | None = None,
    **acquire_kwargs: object,
) -> ScheduleAcquireResult:
    """Acquire one node row through ``Schedule`` / ``ProgramBuilder``.

    Device setup and cfg lowering intentionally stay outside this helper. Only
    program build/acquire is retried, matching the node-level safety knob.
    """
    signal_buffer = SignalBuffer(
        signal_shape,
        dtype=dtype,
        on_update=on_update,
        update_interval=None,
    )
    label = progress_label or (
        f"{env.node_name or 'node'} flux {env.flux_idx + 1} rounds"
    )
    with Schedule(cfg, signal_buffer) as sched:
        builder = sched.prog_builder(
            env.soc,
            env.soccfg,
            cfg=cfg,
            program_cls=program_cls,
        )
        configure_builder(builder)
        signal = builder.build_and_acquire(
            raw2signal_fn=raw2signal_fn,
            retry=acquire_retry(env),
            progress=False,
            progress_label=label,
            progress_leave=False,
            stop_checkers=stop_checkers,
            **acquire_kwargs,
        )
        outcome = sched.outcome

    if outcome.status == "completed":
        return ScheduleAcquireResult(signal=np.asarray(signal))
    if outcome.status == "stopped":
        return ScheduleAcquireResult(signal=None, stopped=True)
    if outcome.status == "failed":
        reason = outcome.reason or "Schedule acquire failed"
        raise RuntimeError(reason) from outcome.exception
    if outcome.status == "interrupted":
        reason = outcome.reason or "Schedule acquire interrupted"
        raise RuntimeError(reason) from outcome.exception
    raise RuntimeError(f"unsupported Schedule outcome: {outcome.status!r}")


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


def is_trusted_decay_scalar_fit(
    real: NDArray[np.float64],
    fit_curve: NDArray[np.float64],
    fit_scalar: float,
    sweep_axis: NDArray[np.float64],
) -> bool:
    """Whether a T1/T2 scalar fit may be fed back downstream.

    The legacy autofluxdep runner used a stricter success gate for scalar decay
    feedback than qubit-frequency fits: residual below 10% of the fit span and a
    fitted time no larger than twice the measured sweep window.
    """
    scalar = float(fit_scalar)
    if not np.isfinite(scalar) or scalar <= 0.0:
        return False

    axis = np.asarray(sweep_axis, dtype=np.float64)
    if axis.size == 0 or not np.all(np.isfinite(axis)):
        return False
    upper = float(np.max(axis))
    if not np.isfinite(upper) or upper <= 0.0:
        return False
    if scalar > _DECAY_SCALAR_MAX_SWEEP_FACTOR * upper:
        return False

    return is_good_fit(real, fit_curve, threshold=_DECAY_SCALAR_RESIDUAL_RATIO)


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
    ``snr`` mirrors the same round for the GUI liveplot title.
    """

    value: NDArray[np.complex128] | None = None
    snr: float = np.nan


def earlystop_snr(schema: NodeCfgSchema, md: Any = None) -> float | None:
    """The SNR early-stop threshold from a Node's schema, or None (no early-stop).

    Mirrors the lower-layer task's ``earlystop_snr``. The knob is an optional
    ``FloatSpec``: unset → the lowered dict omits the key → None (no early-stop).
    A node type without the knob (ro_optimize) likewise yields None. No text
    parsing — the schema already lowered it to ``float | None``.
    """
    return schema.lower(None, md=md).get("earlystop_snr")


def build_stop_checkers(
    env: RunEnv,
    probe: SnrProbe,
    signal2real_fn: Callable[[np.ndarray], np.ndarray],
) -> list[Callable[[], bool]]:
    """The ``stop_checkers`` list every measurement acquire passes to ``.acquire``.

    Threads the run's cooperative cancel (``env.should_stop``) + the SNR early-stop
    (``snr_checker`` reading the running average via ``probe`` once the first
    ``round_hook`` has populated it) — exactly the lower layer's
    ``stop_checkers=[ctx.is_stop, snr_checker(ctx, ...)]`` after data exists."""
    checkers: list[Callable[[], bool]] = []
    if env.should_stop is not None:
        checkers.append(env.should_stop)
    threshold = earlystop_snr(env.schema, md=env.md)
    if threshold is not None:
        check_snr = snr_checker(probe, threshold, signal2real_fn)

        def check_snr_when_ready() -> bool:
            if probe.value is None:
                return False
            return check_snr()

        checkers.append(check_snr_when_ready)
    return checkers


def make_signal_update(
    result: Any,
    idx: int,
    signal2real_fn: Callable[[NDArray[np.complex128]], NDArray[np.float64]],
    round_hook: Callable[[int], None] | None,
    probe: SnrProbe | None = None,
) -> Callable[[NDArray[Any]], None]:
    """Build the ``SignalBuffer`` update hook for Schedule-backed 1-D nodes."""

    def on_update(signal_value: NDArray[Any]) -> None:
        profile_start = perf_now()
        signal = np.asarray(signal_value, dtype=np.complex128)
        real = signal2real_fn(signal)
        if probe is not None:
            probe.value = signal
            probe.snr = estimate_snr(real)
            snr = getattr(result, "snr", None)
            if snr is not None:
                snr[idx] = probe.snr
        np.copyto(result.signal[idx], real)
        if round_hook is not None:
            round_hook(idx)
        _ROUND_HOOK_PERF.record(
            elapsed_ms(profile_start),
            detail=f"idx={idx}",
        )

    return on_update


def fill_decay_fit_or_skip(
    result: Any,
    idx: int,
    real: NDArray[np.float64],
    sweep_axis: NDArray[np.float64],
    fit_scalar: float,
    fit_curve: NDArray[np.float64],
    round_hook: Callable[[int], None] | None,
    logger: Any,
    node_name: str,
) -> bool:
    """Gate a single-scalar 1-D fit and fill (or skip) its Result row.

    Shared by the scalar decay nodes (t1/t2ramsey/t2echo): an SNR-trough or
    unbounded flux point is rejected, the raw row stays shown with nan fit fields,
    and the caller returns a partial ``Patch()``. On a trusted fit, record the
    scalar + curve and fire the round_hook, returning True so the caller builds its
    Patch. The node-specific success log (and any extra Patch keys/modules) stays
    in the caller; only the gate + row-fill bookkeeping is shared here."""
    if not is_trusted_decay_scalar_fit(real, fit_curve, fit_scalar, sweep_axis):
        logger.debug("%s fit @flux%d: untrusted scalar fit — discarded", node_name, idx)
        if round_hook is not None:
            round_hook(idx)  # raw row already shown; fit fields stay nan
        return False
    result.fit_value[idx] = float(fit_scalar)
    np.copyto(result.fit_curve[idx], np.asarray(fit_curve, dtype=np.float64))
    if round_hook is not None:
        round_hook(idx)
    return True

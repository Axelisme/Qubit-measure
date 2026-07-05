"""ro_optimize — 2D readout optimisation by SNR landscape argmax.

Sets this flux point's value on the picked flux device, runs a real readout
acquire over a freq × gain grid (against the flux-aware MockSoc offline or real
hardware), finds the optimum via ``argmax`` (no fit — the peak location IS the
result), fills its Sweep2DResult row in place, and returns a Patch with
``best_ro_freq`` and ``best_ro_gain``, plus the ``opt_readout`` module constructed
from them.

- requires the ``pi_pulse`` module (a pi-pulse is needed to prepare the excited
  state before measuring readout fidelity); the resolver skips until a concrete
  pulse is produced or available in ModuleLibrary.
- reads optional ``best_ro_freq`` and ``best_ro_gain`` (raw prev-point values —
  no smoothing flag: the tracking loop deliberately follows the actual last best
  to plant the Gaussian centre so the optimum tracks across flux points), with
  sensible MHz defaults when absent.
- reads optional ``t1`` (smoothed prev-point T1) for the relax_delay (3·T1) and
  to exercise the dependency mechanism.
- the ``readout`` module is optional (a base readout template); it is the readout
  the cfg sweeps over (its freq/gain are swept), mirroring the real experiment.

No fit step: the 2D landscape is computed in one shot (one effective "round"), so
``round_hook`` is called exactly once after filling the row.

``produce`` lowers the active context + this point's snapshot into a runnable
``RoOptimizeCfgTemplate`` via the Builder's ``make_cfg`` (Fast Fail if the context
is unconfigured — a real acquire needs a concrete ``pi_pulse`` + ``readout``). The
cfg's ``freq_range`` / ``gain_range`` (centred on the previous best, mirroring the
notebook ``RO_OptTask`` ``cfg_maker``) define the swept grid; the acquire runs the
real readout program over it. The mock path uses the flux-aware MockSoc, so the
fidelity landscape varies with the operating flux.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.utils import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.experiment.v2_gui.adapters.twotone.ro_optimize.freq_gain import (
    RoOptFreqGainAdapter,
)
from zcu_tools.gui.app.autofluxdep.cfg import (
    FloatSpec,
    OverridePath,
    OverridePlan,
    SweepValue,
    module_leaf_patches,
    str_choice_spec,
)
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    axis_to_sweep,
    require_flux_device,
    round_progress,
    set_flux_by_name,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.defaults import (
    PULSE_MODULE_LEAF_PATHS,
    READOUT_PULSE_MODULE_LEAF_PATHS,
    adapter_node_schema,
    ctx_md_float,
    ctx_module,
    generation_field,
    readout_pulse_freq,
    readout_pulse_gain,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import (
    PI_PULSE_LIBRARY_ALIASES,
    READOUT_LIBRARY_ALIASES,
)
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Landscape2DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep2DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.profiling import PerfStats, elapsed_ms, perf_now
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.process import smooth_signal_nd

logger = logging.getLogger(__name__)
_RO_LANDSCAPE_PERF = PerfStats("worker.ro_optimize.landscape", logger, slow_ms=20.0)
_RO_PROGRAM_BUILD_PERF = PerfStats(
    "worker.ro_optimize.program_build", logger, slow_ms=50.0
)
_RO_ACQUIRE_PERF = PerfStats("worker.ro_optimize.acquire", logger, slow_ms=50.0)
_RO_MIN_LANDSCAPE_SPAN = 1e-12
_RO_MIN_LOCAL_PROMINENCE_FRACTION = 1e-3


@dataclass(frozen=True)
class _RoOptimum:
    freq_index: int
    gain_index: int
    freq: float
    gain: float


def _ro_signal2real(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Smooth the SNR landscape before argmax (lower-layer ``ro_opt_signal2real``)."""
    return np.abs(smooth_signal_nd(signals, method="wavelet", sigma=1.0))


def _ro_landscape(
    tracker: MomentTracker, shape: tuple[int, ...], *, skew_penalty: float
) -> NDArray[np.float64]:
    """The smoothed (n_freq, n_gain) SNR landscape from the readout-channel tracker.

    The sweep is ``[("ge", 2), ("freq", n), ("gain", m)]`` plus a soft-average
    axis, so the tracker mean carries a leading reps singleton and the ge axis at
    index 1 — same layout the measure-side ``freq_gain`` ro_optimize reduces with
    ``ge_axis=1``. ``snr_as_signal`` reduces the ge axis to a
    per-(freq, gain) SNR; we reshape it to the Result's row shape (dropping the
    singleton) and smooth."""
    profile_start = perf_now()
    snr_start = perf_now()
    snr = snr_as_signal([tracker], ge_axis=1, skew_penalty=skew_penalty)
    snr_ms = elapsed_ms(snr_start)
    smooth_start = perf_now()
    landscape = _ro_signal2real(np.asarray(snr, dtype=np.float64).reshape(shape))
    smooth_ms = elapsed_ms(smooth_start)
    _RO_LANDSCAPE_PERF.record(
        elapsed_ms(profile_start),
        detail=f"shape={shape} snr_ms={snr_ms:.1f} smooth_ms={smooth_ms:.1f}",
    )
    return landscape


def _accepted_ro_optimum(
    landscape: NDArray[np.float64],
    freqs: NDArray[np.float64],
    gains: NDArray[np.float64],
) -> _RoOptimum | None:
    """Return a trustworthy interior readout optimum, or None for fallback.

    The SNR map is already smoothed before it reaches this helper. A feedback point
    is accepted only when the smoothed landscape has finite structure, a unique
    interior maximum, and that maximum rises above its local neighbourhood.
    """
    snr = np.asarray(landscape, dtype=np.float64)
    freq_axis = np.asarray(freqs, dtype=np.float64)
    gain_axis = np.asarray(gains, dtype=np.float64)
    if snr.shape != (freq_axis.size, gain_axis.size):
        raise ValueError(
            "ro_optimize landscape shape does not match axes: "
            f"{snr.shape} vs ({freq_axis.size}, {gain_axis.size})"
        )
    if not np.all(np.isfinite(freq_axis)) or not np.all(np.isfinite(gain_axis)):
        return None
    if freq_axis.size < 3 or gain_axis.size < 3:
        return None

    finite = np.isfinite(snr)
    if not np.any(finite):
        return None
    finite_values = snr[finite]
    span = float(np.max(finite_values) - np.min(finite_values))
    if not np.isfinite(span) or span <= _RO_MIN_LANDSCAPE_SPAN:
        return None

    masked = np.where(finite, snr, -np.inf)
    best_value = float(np.max(masked))
    best_locations = np.argwhere(masked == best_value)
    if best_locations.shape[0] != 1:
        return None
    best_fi = int(best_locations[0, 0])
    best_gi = int(best_locations[0, 1])
    if best_fi in (0, freq_axis.size - 1) or best_gi in (0, gain_axis.size - 1):
        return None

    local = masked[best_fi - 1 : best_fi + 2, best_gi - 1 : best_gi + 2].copy()
    local[1, 1] = -np.inf
    local_neighbors = local[np.isfinite(local)]
    if local_neighbors.size == 0:
        return None
    local_prominence = best_value - float(np.max(local_neighbors))
    min_prominence = max(
        _RO_MIN_LANDSCAPE_SPAN,
        _RO_MIN_LOCAL_PROMINENCE_FRACTION * span,
    )
    if local_prominence < min_prominence:
        return None

    return _RoOptimum(
        freq_index=best_fi,
        gain_index=best_gi,
        freq=float(freq_axis[best_fi]),
        gain=float(gain_axis[best_gi]),
    )


# Default axis specs: (start, stop, npts). The readout-frequency fallback follows
# the notebook raw cfg shape: previous-best center ± a small range. A calibrated
# readout or previous ro_optimize result recenters the real run.
_DEFAULT_FREQ_RANGE: tuple[float, float, int] = (5999.0, 6001.0, 21)
_DEFAULT_GAIN_RANGE: tuple[float, float, int] = (0.45, 0.55, 21)

_DEFAULT_CENTER_FREQ = 6000.0  # MHz — baseline readout resonance fallback
_DEFAULT_CENTER_GAIN = 0.5

_DEFAULT_T1 = 10.0  # us — fallback T1 for the relax_delay (3·T1)
_DEFAULT_RELAX_FACTOR = 3.0
_DEFAULT_FREQ_HALF_WIDTH = 1.0
_DEFAULT_GAIN_HALF_WIDTH = 0.05
_RANGE_MODE_PREVIOUS_BEST = "previous_best"
_RANGE_MODE_FIXED = "fixed"
_WINDOW_MODE_DEFAULT_SWEEP = "from_default_sweep"
_WINDOW_MODE_FIXED_HALF_WIDTH = "fixed_half_width"
_RELAX_DELAY_MODE_AUTO_T1 = "auto_t1"
_RELAX_DELAY_MODE_FIXED = "fixed"


class RoOptimizeModuleCfg(ConfigBase):
    """The modules ro_optimize lowers — an optional reset + the pi-pulse + readout.

    Mirrors the lower-layer ``experiment/v2/autofluxdep`` ``RO_OptModuleCfg``: the
    ``pi_pulse`` prepares the excited state and the ``readout`` is the pulse whose
    ``freq`` / ``gain`` the sweep optimises over.
    """

    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: PulseReadoutCfg


class RoOptimizeCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base program cfg ro_optimize lowers a context into.

    ``ProgramV2Cfg`` (reps/rounds/relax) + the ``ExpCfgModel`` device fields, plus
    the ``modules`` (pi_pulse + readout) and the ``freq_range`` / ``gain_range``
    sweep windows — exactly the lower-layer ``RO_OptCfgTemplate``. The flux ``dev``
    entry and the concrete ``freq`` / ``gain`` ``SweepCfg`` are merged in by the
    lower-layer ``run`` (the GUI prototype reads the window centres straight off
    the template to plant the synthetic landscape); ``freq_range`` / ``gain_range``
    are stripped before the runnable ``RO_OptCfg`` is validated downstream.
    """

    modules: RoOptimizeModuleCfg
    freq_range: tuple[float, float]
    gain_range: tuple[float, float]
    skew_penalty: float = Field(default=0.0, ge=0.0)


def _default_t1() -> None:
    return None


def _default_best_freq() -> None:
    return None


def _default_best_gain() -> None:
    return None


def _seed_t1(ctx: Any | None) -> float:
    return ctx_md_float(ctx, "t1") or _DEFAULT_T1


def _seed_readout_freq(ctx: Any | None) -> float:
    md_value = ctx_md_float(ctx, "r_f")
    if md_value is not None:
        return md_value
    module = ctx_module(ctx, "readout_dpm", "readout_rf", "readout", "res_readout")
    return readout_pulse_freq(module) or _DEFAULT_CENTER_FREQ


def _seed_readout_gain(ctx: Any | None) -> float:
    module = ctx_module(ctx, "readout_dpm", "readout_rf", "readout", "res_readout")
    return readout_pulse_gain(module) or _DEFAULT_CENTER_GAIN


def _snapshot_float(snapshot: Snapshot, key: str, fallback: float) -> float:
    value = snapshot.get(key)
    if value is None:
        return fallback
    return float(value)


def _center_of(sweep: Any) -> float:
    return 0.5 * (float(sweep.start) + float(sweep.stop))


def _default_readout() -> Any | None:
    return None


def _resolve_range(
    mode: str,
    *,
    previous: float,
    fixed: Any,
    label: str,
    window_mode: str,
    half_width: float,
) -> tuple[float, float]:
    fixed_start = float(fixed.start)
    fixed_stop = float(fixed.stop)
    if mode == _RANGE_MODE_PREVIOUS_BEST:
        if window_mode == _WINDOW_MODE_DEFAULT_SWEEP:
            width = abs(fixed_stop - fixed_start) / 2.0
        elif window_mode == _WINDOW_MODE_FIXED_HALF_WIDTH:
            width = float(half_width)
        else:
            raise RuntimeError(
                f"unsupported ro_optimize {label} window mode: {window_mode!r}"
            )
        center = float(previous)
        start = center - width
        stop = center + width
    elif mode == _RANGE_MODE_FIXED:
        start = fixed_start
        stop = fixed_stop
    else:
        raise RuntimeError(f"unsupported ro_optimize {label} range mode: {mode!r}")
    if label == "gain":
        return (max(0.0, start), min(1.0, stop))
    return (start, stop)


def _resolve_relax_delay(
    mode: str, *, t1: float, fixed: float, relax_factor: float
) -> float:
    if mode == _RELAX_DELAY_MODE_AUTO_T1:
        return float(relax_factor) * float(t1)
    if mode == _RELAX_DELAY_MODE_FIXED:
        return float(fixed)
    raise RuntimeError(f"unsupported ro_optimize relax_delay_mode: {mode!r}")


def _raw_range_tuple(value: Any) -> tuple[float, float]:
    if hasattr(value, "start") and hasattr(value, "stop"):
        return (float(value.start), float(value.stop))
    lo, hi = value
    return (float(lo), float(hi))


def _pop_sweep_ranges(
    raw_cfg: dict[str, Any], keys: tuple[str, ...]
) -> dict[str, tuple[float, float]]:
    sweep = raw_cfg.pop("sweep", None)
    if not isinstance(sweep, dict):
        raise RuntimeError("ro_optimize raw cfg has no sweep section")
    ranges: dict[str, tuple[float, float]] = {}
    for key in keys:
        if key not in sweep:
            raise RuntimeError(f"ro_optimize raw cfg has no sweep.{key}")
        ranges[key] = _raw_range_tuple(sweep[key])
    return ranges


class RoOptimizeNode(Node):
    """One flux point's ro_optimize: set flux → real acquire → SNR argmax → Patch.

    Mirrors the lower-layer RO optimize Schedule acquire + ``run``: a
    ``ModularProgramV2`` (Reset → ge-Branch(pi_pulse) → PulseReadout) sweeps the
    readout freq × gain (interleaved with the ge axis), a ``MomentTracker``
    accumulates per-shot moments, and ``snr_as_signal`` turns them
    into an SNR landscape whose argmax is the best (freq, gain). No fit step.
    """

    def __init__(self, env: RunEnv, builder: RoOptimizeBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

        result: Sweep2DResult = env.result
        idx = env.flux_idx

        # Lower the active context into the run cfg (Fast Fail if unconfigured: a
        # real acquire needs a concrete pi_pulse + readout pulse).
        cfg = self._builder.make_cfg(env, snapshot)
        freqs = np.linspace(
            float(cfg.freq_range[0]), float(cfg.freq_range[1]), result.n_freq
        )
        gains = np.linspace(
            float(cfg.gain_range[0]), float(cfg.gain_range[1]), result.n_gain
        )
        result.freq[:] = freqs
        result.gain[:] = gains

        flux_device = require_flux_device(env, "ro_optimize")
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

        # Sweep the readout freq × gain over the cfg-derived axes (lower layer sets
        # the freq / gain params on the readout pulse). Keep Result axes in sync so
        # the plotter shows the same range the program actually sweeps.
        freq_sweep = axis_to_sweep(freqs)
        gain_sweep = axis_to_sweep(gains)
        cfg.modules.readout.set_param("freq", sweep2param("freq", freq_sweep))
        cfg.modules.readout.set_param("gain", sweep2param("gain", gain_sweep))

        result.flux[idx] = env.flux

        tracker = MomentTracker()

        with round_progress(cfg.rounds, "ro_optimize", idx) as update_round_progress:

            def on_round(round_count: int, _avg_d: Any) -> None:
                update_round_progress(round_count)
                # the SNR landscape (n_freq, n_gain) accumulated so far → overwrite row
                landscape = _ro_landscape(
                    tracker,
                    result.signal[idx].shape,
                    skew_penalty=cfg.skew_penalty,
                )
                np.copyto(result.signal[idx], landscape)
                if env.round_hook is not None:
                    env.round_hook(idx)

            program_start = perf_now()
            program = ModularProgramV2(
                env.soccfg,
                cfg,
                modules=[
                    Reset("reset", cfg.modules.reset),
                    Branch("ge", [], Pulse("pi_pulse", cfg.modules.pi_pulse)),
                    PulseReadout("readout", cfg.modules.readout),
                ],
                sweep=[
                    ("ge", 2),
                    ("freq", freq_sweep),
                    ("gain", gain_sweep),
                ],
            )
            _RO_PROGRAM_BUILD_PERF.record(
                elapsed_ms(program_start),
                detail=(
                    f"idx={idx} reps={cfg.reps} rounds={cfg.rounds} "
                    f"freq_points={result.n_freq} gain_points={result.n_gain}"
                ),
            )

            acquire_start = perf_now()
            program.acquire(
                env.soc,
                progress=False,
                round_hook=on_round,
                trackers=[tracker],
                stop_checkers=[env.should_stop]
                if env.should_stop is not None
                else None,
            )
            _RO_ACQUIRE_PERF.record(
                elapsed_ms(acquire_start),
                detail=(
                    f"idx={idx} reps={cfg.reps} rounds={cfg.rounds} "
                    f"freq_points={result.n_freq} gain_points={result.n_gain}"
                ),
            )

        landscape = _ro_landscape(
            tracker,
            result.signal[idx].shape,
            skew_penalty=cfg.skew_penalty,
        )
        np.copyto(result.signal[idx], landscape)

        optimum = _accepted_ro_optimum(landscape, freqs, gains)
        if optimum is None:
            logger.debug(
                "ro_optimize @flux%d: untrusted SNR landscape — feedback omitted",
                idx,
            )
            if env.round_hook is not None:
                env.round_hook(idx)
            return Patch()

        best_freq = optimum.freq
        best_gain = optimum.gain

        result.best_freq[idx] = best_freq
        result.best_gain[idx] = best_gain
        if env.round_hook is not None:
            env.round_hook(idx)

        logger.debug(
            "ro_optimize @flux%d: best_freq=%.3f best_gain=%.3f",
            idx,
            best_freq,
            best_gain,
        )

        # produce the tuned readout MODULE so downstream consumers (t1 / t2* / mist)
        # sweep against the optimised point: a deepcopy of the (real) readout cfg
        # with its freq / gain set to the argmax, mirroring the lower-layer
        # ``RO_OptTask.run`` (deepcopy(cfg.modules.readout); set_param freq/gain).
        opt_readout = deepcopy(cfg.modules.readout)
        opt_readout.set_param("freq", best_freq)
        opt_readout.set_param("gain", best_gain)

        patch = Patch()
        patch.set("best_ro_freq", best_freq)
        patch.set("best_ro_gain", best_gain)
        patch.set_module("opt_readout", opt_readout)
        return patch


class RoOptimizeBuilder(Builder):
    """The ro_optimize provider — 2D Gaussian synth, argmax (no fit), overwrite plot.

    Sweeps a freq × gain grid per flux point, synthesises a readout-fidelity
    landscape, and finds the optimum via argmax. No fitting: the Gaussian peak
    location IS the best readout point. Produces ``best_ro_freq``, ``best_ro_gain``,
    and the ``opt_readout`` module for downstream consumers (e.g. t1, mist).
    """

    name = "ro_optimize"
    provides = ("best_ro_freq", "best_ro_gain")
    provides_modules = ("opt_readout",)
    optional = (
        Dependency("t1", smooth="ewma", default=_default_t1),
        Dependency("best_ro_freq", default=_default_best_freq),
        Dependency("best_ro_gain", default=_default_best_gain),
    )
    requires_modules = (ModuleDep("pi_pulse", aliases=PI_PULSE_LIBRARY_ALIASES),)
    optional_modules = (
        ModuleDep("readout", default=_default_readout, aliases=READOUT_LIBRARY_ALIASES),
    )

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Adapter-backed default cfg plus autofluxdep generation controls."""
        t1_seed = _seed_t1(ctx)
        freq_seed = _seed_readout_freq(ctx)
        gain_seed = _seed_readout_gain(ctx)
        return adapter_node_schema(
            RoOptFreqGainAdapter,
            ctx,
            logical_paths={
                "reset": "modules.reset",
                "pi_pulse": "modules.pi_pulse",
                "readout": "modules.readout",
                "relax_delay": "relax_delay",
                "skew_penalty": "skew_penalty",
                "reps": "reps",
                "rounds": "rounds",
                "freq_range": "sweep.freq",
                "gain_range": "sweep.gain",
            },
            generation_fields=(
                generation_field(
                    "freq_range_mode",
                    "freq_range_mode",
                    str_choice_spec(
                        "freq_range_mode",
                        (_RANGE_MODE_PREVIOUS_BEST, _RANGE_MODE_FIXED),
                    ),
                    _RANGE_MODE_PREVIOUS_BEST,
                    group="sweep",
                ),
                generation_field(
                    "gain_range_mode",
                    "gain_range_mode",
                    str_choice_spec(
                        "gain_range_mode",
                        (_RANGE_MODE_PREVIOUS_BEST, _RANGE_MODE_FIXED),
                    ),
                    _RANGE_MODE_PREVIOUS_BEST,
                    group="sweep",
                ),
                generation_field(
                    "relax_delay_mode",
                    "relax_delay_mode",
                    str_choice_spec(
                        "relax_delay_mode",
                        (_RELAX_DELAY_MODE_AUTO_T1, _RELAX_DELAY_MODE_FIXED),
                    ),
                    _RELAX_DELAY_MODE_AUTO_T1,
                    group="timing",
                ),
                generation_field(
                    "t1_seed_us",
                    "t1_seed_us",
                    FloatSpec(label="t1_seed_us"),
                    t1_seed,
                    group="timing",
                ),
                generation_field(
                    "relax_factor",
                    "relax_factor",
                    FloatSpec(label="relax_factor"),
                    _DEFAULT_RELAX_FACTOR,
                    group="timing",
                ),
                generation_field(
                    "freq_window_mode",
                    "freq_window_mode",
                    str_choice_spec(
                        "freq_window_mode",
                        (_WINDOW_MODE_FIXED_HALF_WIDTH, _WINDOW_MODE_DEFAULT_SWEEP),
                    ),
                    _WINDOW_MODE_FIXED_HALF_WIDTH,
                    group="feedback",
                ),
                generation_field(
                    "freq_half_width_mhz",
                    "freq_half_width_mhz",
                    FloatSpec(label="freq_half_width_mhz"),
                    _DEFAULT_FREQ_HALF_WIDTH,
                    group="feedback",
                ),
                generation_field(
                    "gain_window_mode",
                    "gain_window_mode",
                    str_choice_spec(
                        "gain_window_mode",
                        (_WINDOW_MODE_FIXED_HALF_WIDTH, _WINDOW_MODE_DEFAULT_SWEEP),
                    ),
                    _WINDOW_MODE_FIXED_HALF_WIDTH,
                    group="feedback",
                ),
                generation_field(
                    "gain_half_width",
                    "gain_half_width",
                    FloatSpec(label="gain_half_width"),
                    _DEFAULT_GAIN_HALF_WIDTH,
                    group="feedback",
                ),
            ),
            default_overrides={
                "reps": 1000,
                "rounds": 10,
                "relax_delay": _DEFAULT_RELAX_FACTOR * t1_seed,
                "freq_range": SweepValue(
                    freq_seed - _DEFAULT_FREQ_HALF_WIDTH,
                    freq_seed + _DEFAULT_FREQ_HALF_WIDTH,
                    expts=31,
                ),
                "gain_range": SweepValue(
                    max(0.0, gain_seed - _DEFAULT_GAIN_HALF_WIDTH),
                    min(1.0, gain_seed + _DEFAULT_GAIN_HALF_WIDTH),
                    expts=31,
                ),
            },
            path_renames={"modules.qub_pulse": "modules.pi_pulse"},
        )

    def make_init_result(
        self, schema: NodeCfgSchema, flux: Any, md: Any = None
    ) -> Sweep2DResult:
        knobs = schema.lower(None, md=md)
        freq_range_knob = knobs["freq_range"]
        gain_range_knob = knobs["gain_range"]
        freq_expts = int(freq_range_knob.expts)
        gain_expts = int(gain_range_knob.expts)

        # The first allocation has no per-flux snapshot yet, so previous-best modes
        # use the default centres; fixed modes use the operator's explicit ranges.
        # ``produce`` rebuilds these axes from the lowered per-point cfg before
        # acquiring.
        freq_range = _resolve_range(
            str(knobs["freq_range_mode"]),
            previous=_center_of(freq_range_knob),
            fixed=freq_range_knob,
            label="freq",
            window_mode=str(knobs["freq_window_mode"]),
            half_width=float(knobs["freq_half_width_mhz"]),
        )
        gain_range = _resolve_range(
            str(knobs["gain_range_mode"]),
            previous=_center_of(gain_range_knob),
            fixed=gain_range_knob,
            label="gain",
            window_mode=str(knobs["gain_window_mode"]),
            half_width=float(knobs["gain_half_width"]),
        )
        freqs = np.linspace(freq_range[0], freq_range[1], freq_expts)
        gains = np.linspace(gain_range[0], gain_range[1], gain_expts)
        return Sweep2DResult.allocate(flux, freqs, gains)

    def make_plotter(self, figure: Any) -> Landscape2DPlotter:
        return Landscape2DPlotter(figure, title="ro_optimize")

    def build_node(self, env: RunEnv) -> RoOptimizeNode:
        return RoOptimizeNode(env, self)

    def override_plan(self, schema: NodeCfgSchema) -> OverridePlan:
        knobs = schema.read_knobs()
        paths: list[OverridePath] = []
        paths.extend(
            OverridePath(
                f"modules.pi_pulse.{leaf}",
                "all_points",
                "pi_pulse module dependency",
                "pi pulse is resolved from workflow/module-library dependency",
            )
            for leaf in PULSE_MODULE_LEAF_PATHS
        )
        paths.extend(
            OverridePath(
                f"modules.readout.{leaf}",
                "all_points",
                "readout module dependency",
                "readout module is resolved from workflow/module-library dependency",
            )
            for leaf in READOUT_PULSE_MODULE_LEAF_PATHS
        )
        if knobs.get("relax_delay_mode") == _RELAX_DELAY_MODE_AUTO_T1:
            paths.append(
                OverridePath(
                    "relax_delay",
                    "all_points",
                    "generation.timing.relax_delay_mode",
                    "relax delay is generated from T1 feedback",
                )
            )
        if knobs.get("freq_range_mode") == _RANGE_MODE_PREVIOUS_BEST:
            paths.append(
                OverridePath(
                    "sweep.freq",
                    "all_points",
                    "generation.sweep.freq_range_mode",
                    "readout frequency window is generated from previous best",
                )
            )
        if knobs.get("gain_range_mode") == _RANGE_MODE_PREVIOUS_BEST:
            paths.append(
                OverridePath(
                    "sweep.gain",
                    "all_points",
                    "generation.sweep.gain_range_mode",
                    "readout gain window is generated from previous best",
                )
            )
        return OverridePlan(tuple(paths))

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> RoOptimizeCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the notebook's ro_optimize ``cfg_maker`` (runs in ``produce``,
        where the snapshot is available): the ``pi_pulse`` and ``readout`` modules
        come whole from the snapshot (lenrabi produces the pi-pulse; the readout is
        the base template), the relax_delay can be auto-derived from ``3·T1``, and
        the ``freq_range`` / ``gain_range`` can be either fixed ranges from the
        Default cfg or those same range widths re-centered around the previous best.
        The flux ``dev`` entry and the concrete ``freq`` / ``gain`` sweeps are NOT
        here — the lower-layer ``run`` merges them.

        Raises if the ml is unavailable or the pi_pulse / readout modules are
        unset — a real run needs both concrete modules (Fast Fail), unlike the
        synthetic path which fabricates a landscape.
        """
        ml = env.ml
        if ml is None:
            raise RuntimeError("ro_optimize.make_cfg needs an active ModuleLibrary")
        pi_pulse = snapshot.module("pi_pulse")
        readout = snapshot.module("readout")
        if pi_pulse is None or readout is None:
            raise RuntimeError(
                "ro_optimize.make_cfg needs the pi_pulse + readout modules "
                "(none produced or preset)"
            )
        knobs = env.knobs()
        freq_range_knob = knobs["freq_range"]
        gain_range_knob = knobs["gain_range"]
        freq_range = _resolve_range(
            str(knobs["freq_range_mode"]),
            previous=_snapshot_float(
                snapshot, "best_ro_freq", _center_of(freq_range_knob)
            ),
            fixed=freq_range_knob,
            label="freq",
            window_mode=str(knobs["freq_window_mode"]),
            half_width=float(knobs["freq_half_width_mhz"]),
        )
        gain_range = _resolve_range(
            str(knobs["gain_range_mode"]),
            previous=_snapshot_float(
                snapshot, "best_ro_gain", _center_of(gain_range_knob)
            ),
            fixed=gain_range_knob,
            label="gain",
            window_mode=str(knobs["gain_window_mode"]),
            half_width=float(knobs["gain_half_width"]),
        )
        t1 = _snapshot_float(snapshot, "t1", float(knobs["t1_seed_us"]))
        relax_delay = _resolve_relax_delay(
            str(knobs["relax_delay_mode"]),
            t1=t1,
            fixed=float(knobs["relax_delay"]),
            relax_factor=float(knobs["relax_factor"]),
        )
        patches: dict[str, object] = {}
        patches.update(
            module_leaf_patches(
                prefix="modules.pi_pulse",
                module=pi_pulse,
                leaf_paths=PULSE_MODULE_LEAF_PATHS,
            )
        )
        patches.update(
            module_leaf_patches(
                prefix="modules.readout",
                module=readout,
                leaf_paths=READOUT_PULSE_MODULE_LEAF_PATHS,
            )
        )
        if str(knobs["relax_delay_mode"]) == _RELAX_DELAY_MODE_AUTO_T1:
            patches["relax_delay"] = relax_delay
        if str(knobs["freq_range_mode"]) == _RANGE_MODE_PREVIOUS_BEST:
            patches["sweep.freq"] = freq_range
        if str(knobs["gain_range_mode"]) == _RANGE_MODE_PREVIOUS_BEST:
            patches["sweep.gain"] = gain_range
        raw_cfg = self.point_cfg(env, patches)
        ranges = _pop_sweep_ranges(raw_cfg, ("freq", "gain"))
        raw_cfg["freq_range"] = ranges["freq"]
        raw_cfg["gain_range"] = ranges["gain"]
        return ml.make_cfg(raw_cfg, RoOptimizeCfgTemplate)

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

No fit step: Schedule acquire updates the SNR landscape row as tracker data arrives;
after acquire, ``produce`` refreshes the row with the final landscape and then emits
the optimum or fallback fields.

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
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.gui.app.autofluxdep.cfg import OverridePlan
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    DEFAULT_ACQUIRE_RETRY,
    acquire_retry,
    axis_to_sweep,
    require_flux_device,
    set_flux_by_name,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.dependency_defaults import (
    missing_info_value,
    missing_module_value,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import (
    PI_PULSE_LIBRARY_ALIASES,
    READOUT_LIBRARY_ALIASES,
)
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Landscape2DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.readout_defaults import (
    seed_readout_freq,
    seed_readout_gain,
)
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep2DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.timing_defaults import (
    auto_relax_delay_from_t1,
    seed_md_float,
    snapshot_float,
)
from zcu_tools.gui.app.autofluxdep.nodes.utils import (
    NodeOverridePlan,
    NodeSchemaBuilder,
)
from zcu_tools.gui.app.autofluxdep.nodes.utils.override_plan import (
    pulse_module_patches,
    readout_module_patches,
)
from zcu_tools.gui.app.autofluxdep.nodes.utils.timing import pop_sweep_ranges
from zcu_tools.gui.app.autofluxdep.profiling import PerfStats, elapsed_ms, perf_now
from zcu_tools.gui.cfg import CenteredSweepValue
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadout,
    PulseReadoutCfg,
    sweep2param,
)
from zcu_tools.utils.process import smooth_signal_nd

logger = logging.getLogger(__name__)
_RO_LANDSCAPE_PERF = PerfStats("worker.ro_optimize.landscape", logger, slow_ms=20.0)
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


_RANGE_MODE_PREVIOUS_BEST = "previous_best"
_RANGE_MODE_FIXED = "fixed"
_WINDOW_MODE_DEFAULT_SWEEP = "from_default_sweep"
_WINDOW_MODE_FIXED_HALF_WIDTH = "fixed_half_width"
_RELAX_DELAY_MODE_AUTO_T1 = "auto_t1"
_RELAX_DELAY_MODE_FIXED = "fixed"


class RoOptimizeModuleCfg(ConfigBase):
    """The modules ro_optimize lowers — the pi-pulse + readout.

    Mirrors the lower-layer ``experiment/v2/autofluxdep`` ``RO_OptModuleCfg``: the
    ``pi_pulse`` prepares the excited state and the ``readout`` is the pulse whose
    ``freq`` / ``gain`` the sweep optimises over.
    """

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


def _center_of(sweep: Any) -> float:
    return 0.5 * (float(sweep.start) + float(sweep.stop))


def _centered_window(start: float, stop: float, *, expts: int) -> CenteredSweepValue:
    lo = float(start)
    hi = float(stop)
    return CenteredSweepValue(center=0.5 * (lo + hi), span=abs(hi - lo), expts=expts)


class RoOptimizeNode(Node):
    """One flux point's ro_optimize: set flux → real acquire → SNR argmax → Patch.

    Mirrors the lower-layer RO optimize Schedule acquire + ``run``: a
    ``ModularProgramV2`` (ge-Branch(pi_pulse) → PulseReadout) sweeps the
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

        def tracker_signal(_raw: Any) -> NDArray[np.float64]:
            return _ro_landscape(
                tracker,
                result.signal[idx].shape,
                skew_penalty=cfg.skew_penalty,
            )

        def on_update(landscape_value: NDArray[Any]) -> None:
            np.copyto(
                result.signal[idx],
                np.asarray(landscape_value, dtype=np.float64),
            )
            if env.round_hook is not None:
                env.round_hook(idx)

        acquire_start = perf_now()
        signal_buffer = SignalBuffer(
            result.signal[idx].shape,
            dtype=np.float64,
            on_update=on_update,
            update_interval=None,
        )
        with Schedule(cfg, signal_buffer) as sched:
            builder = sched.prog_builder(
                env.soc,
                env.soccfg,
                cfg=cfg,
                program_cls=ModularProgramV2,
            )
            builder.add(
                [
                    Branch("ge", [], Pulse("pi_pulse", cfg.modules.pi_pulse)),
                    PulseReadout("readout", cfg.modules.readout),
                ]
            ).declare_sweep("ge", 2).declare_sweep("freq", freq_sweep).declare_sweep(
                "gain", gain_sweep
            )
            signal = builder.build_and_acquire(
                raw2signal_fn=tracker_signal,
                retry=acquire_retry(env),
                progress=False,
                progress_label=f"{env.node_name or 'ro_optimize'} flux {idx + 1} rounds",
                progress_leave=False,
                trackers=[tracker],
            )
            outcome = sched.outcome
        _RO_ACQUIRE_PERF.record(
            elapsed_ms(acquire_start),
            detail=(
                f"idx={idx} reps={cfg.reps} rounds={cfg.rounds} "
                f"freq_points={result.n_freq} gain_points={result.n_gain}"
            ),
        )
        if outcome.status == "stopped":
            return Patch()
        if outcome.status == "failed":
            reason = outcome.reason or "ro_optimize Schedule acquire failed"
            raise RuntimeError(reason) from outcome.exception
        if outcome.status == "interrupted":
            reason = outcome.reason or "ro_optimize Schedule acquire interrupted"
            raise RuntimeError(reason) from outcome.exception
        if outcome.status != "completed":
            raise RuntimeError(
                f"unsupported ro_optimize Schedule outcome: {outcome.status!r}"
            )

        landscape = np.asarray(signal, dtype=np.float64)
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
        Dependency("t1", smooth="ewma", default=missing_info_value),
        Dependency("best_ro_freq", default=missing_info_value),
        Dependency("best_ro_gain", default=missing_info_value),
    )
    requires_modules = (ModuleDep("pi_pulse", aliases=PI_PULSE_LIBRARY_ALIASES),)
    optional_modules = (
        ModuleDep(
            "readout", default=missing_module_value, aliases=READOUT_LIBRARY_ALIASES
        ),
    )

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Default cfg plus autofluxdep generation controls."""
        t1_seed = seed_md_float(ctx, "t1", 10.0)
        relax_factor = 3.0
        freq_half_width_mhz = 1.0
        gain_half_width = 0.1
        freq_seed = seed_readout_freq(ctx, 6000.0)
        gain_seed = seed_readout_gain(ctx, 0.1)

        return (
            NodeSchemaBuilder(ctx, label="Readout Optimize")
            .pulse(
                "pi_pulse",
                "modules.pi_pulse",
                label="Pi Pulse",
                library_keys=PI_PULSE_LIBRARY_ALIASES,
            )
            .pulse_readout(
                "readout",
                "modules.readout",
                label="Readout",
                library_keys=READOUT_LIBRARY_ALIASES,
                locked={
                    "pulse_cfg.freq": 0.0,
                    "ro_cfg.ro_freq": 0.0,
                    "pulse_cfg.gain": 0.0,
                },
            )
            .float(
                "relax_delay",
                "relax_delay",
                label="Relax delay (us)",
                default=auto_relax_delay_from_t1(
                    t1_seed,
                    factor=relax_factor,
                    minimum=None,
                ),
                decimals=3,
            )
            .float(
                "skew_penalty",
                "skew_penalty",
                label="Skew penalty",
                default=0.0,
                decimals=3,
            )
            .int("reps", "reps", label="Reps", default=1000)
            .int("rounds", "rounds", label="Rounds", default=10)
            .centered_sweep(
                "freq_range",
                "sweep.freq",
                label="Freq (MHz)",
                default=_centered_window(
                    freq_seed - freq_half_width_mhz,
                    freq_seed + freq_half_width_mhz,
                    expts=31,
                ),
                tooltip=(
                    "Readout frequency search window; stored as center/span "
                    "and lowered to start/stop for the program."
                ),
            )
            .centered_sweep(
                "gain_range",
                "sweep.gain",
                label="Gain",
                default=_centered_window(
                    max(0.0, gain_seed - gain_half_width),
                    min(1.0, gain_seed + gain_half_width),
                    expts=31,
                ),
                tooltip=(
                    "Readout gain search window; stored as center/span and "
                    "lowered to start/stop for the program."
                ),
            )
            .acquisition(retry=DEFAULT_ACQUIRE_RETRY)
            .choice(
                "relax_delay_mode",
                "generation.relax.relax_delay_mode",
                label="delay_mode",
                choices=(_RELAX_DELAY_MODE_AUTO_T1, _RELAX_DELAY_MODE_FIXED),
                default=_RELAX_DELAY_MODE_AUTO_T1,
                tooltip="Auto derives relax delay from T1; fixed keeps Default cfg delay.",
            )
            .float(
                "t1_seed_us",
                "generation.relax.t1_seed_us",
                label="initial_t1_us",
                default=t1_seed,
                tooltip="Initial T1 before measured feedback exists.",
            )
            .float(
                "relax_factor",
                "generation.relax.relax_factor",
                label="factor",
                default=relax_factor,
                tooltip="Multiplier applied to T1 for auto relax delay.",
            )
            .choice_fields(
                "generation.relax",
                "relax_delay_mode",
                {
                    _RELAX_DELAY_MODE_FIXED: (),
                    _RELAX_DELAY_MODE_AUTO_T1: ("t1_seed_us", "relax_factor"),
                },
            )
            .choice(
                "freq_range_mode",
                "generation.freq_search.freq_range_mode",
                label="freq_mode",
                choices=(_RANGE_MODE_PREVIOUS_BEST, _RANGE_MODE_FIXED),
                default=_RANGE_MODE_PREVIOUS_BEST,
                tooltip="Previous-best recenters readout frequency search each flux.",
            )
            .choice(
                "freq_window_mode",
                "generation.freq_search.freq_window_mode",
                label="freq_mode",
                choices=(_WINDOW_MODE_FIXED_HALF_WIDTH, _WINDOW_MODE_DEFAULT_SWEEP),
                default=_WINDOW_MODE_FIXED_HALF_WIDTH,
                tooltip="Choose fixed half-width or the Default cfg frequency sweep.",
            )
            .float(
                "freq_half_width_mhz",
                "generation.freq_search.freq_half_width_mhz",
                label="freq_half_width_mhz",
                default=freq_half_width_mhz,
                tooltip="Frequency half-width around the readout search center.",
            )
            .choice_fields(
                "generation.freq_search",
                "freq_range_mode",
                {
                    _RANGE_MODE_FIXED: (),
                    _RANGE_MODE_PREVIOUS_BEST: (
                        "freq_window_mode",
                        "freq_half_width_mhz",
                    ),
                },
            )
            .choice_fields(
                "generation.freq_search",
                "freq_window_mode",
                {
                    _WINDOW_MODE_DEFAULT_SWEEP: (),
                    _WINDOW_MODE_FIXED_HALF_WIDTH: ("freq_half_width_mhz",),
                },
            )
            .choice(
                "gain_range_mode",
                "generation.gain_search.gain_range_mode",
                label="gain_mode",
                choices=(_RANGE_MODE_PREVIOUS_BEST, _RANGE_MODE_FIXED),
                default=_RANGE_MODE_PREVIOUS_BEST,
                tooltip="Previous-best recenters readout gain search each flux.",
            )
            .choice(
                "gain_window_mode",
                "generation.gain_search.gain_window_mode",
                label="gain_mode",
                choices=(_WINDOW_MODE_FIXED_HALF_WIDTH, _WINDOW_MODE_DEFAULT_SWEEP),
                default=_WINDOW_MODE_FIXED_HALF_WIDTH,
                tooltip="Choose fixed half-width or the Default cfg gain sweep.",
            )
            .float(
                "gain_half_width",
                "generation.gain_search.gain_half_width",
                label="gain_half_width",
                default=gain_half_width,
                tooltip="Gain half-width around the readout search center.",
            )
            .choice_fields(
                "generation.gain_search",
                "gain_range_mode",
                {
                    _RANGE_MODE_FIXED: (),
                    _RANGE_MODE_PREVIOUS_BEST: (
                        "gain_window_mode",
                        "gain_half_width",
                    ),
                },
            )
            .choice_fields(
                "generation.gain_search",
                "gain_window_mode",
                {
                    _WINDOW_MODE_DEFAULT_SWEEP: (),
                    _WINDOW_MODE_FIXED_HALF_WIDTH: ("gain_half_width",),
                },
            )
            .build()
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
        freq_fixed_start = float(freq_range_knob.start)
        freq_fixed_stop = float(freq_range_knob.stop)
        freq_range_mode = str(knobs["freq_range_mode"])
        if freq_range_mode == _RANGE_MODE_PREVIOUS_BEST:
            freq_window_mode = str(knobs["freq_window_mode"])
            if freq_window_mode == _WINDOW_MODE_DEFAULT_SWEEP:
                freq_width = abs(freq_fixed_stop - freq_fixed_start) / 2.0
            elif freq_window_mode == _WINDOW_MODE_FIXED_HALF_WIDTH:
                freq_width = float(knobs["freq_half_width_mhz"])
            else:
                raise RuntimeError(
                    f"unsupported ro_optimize freq window mode: {freq_window_mode!r}"
                )
            freq_center = _center_of(freq_range_knob)
            freq_range = (freq_center - freq_width, freq_center + freq_width)
        elif freq_range_mode == _RANGE_MODE_FIXED:
            freq_range = (freq_fixed_start, freq_fixed_stop)
        else:
            raise RuntimeError(
                f"unsupported ro_optimize freq range mode: {freq_range_mode!r}"
            )

        gain_fixed_start = float(gain_range_knob.start)
        gain_fixed_stop = float(gain_range_knob.stop)
        gain_range_mode = str(knobs["gain_range_mode"])
        if gain_range_mode == _RANGE_MODE_PREVIOUS_BEST:
            gain_window_mode = str(knobs["gain_window_mode"])
            if gain_window_mode == _WINDOW_MODE_DEFAULT_SWEEP:
                gain_width = abs(gain_fixed_stop - gain_fixed_start) / 2.0
            elif gain_window_mode == _WINDOW_MODE_FIXED_HALF_WIDTH:
                gain_width = float(knobs["gain_half_width"])
            else:
                raise RuntimeError(
                    f"unsupported ro_optimize gain window mode: {gain_window_mode!r}"
                )
            gain_center = _center_of(gain_range_knob)
            gain_range = (
                max(0.0, gain_center - gain_width),
                min(1.0, gain_center + gain_width),
            )
        elif gain_range_mode == _RANGE_MODE_FIXED:
            gain_range = (
                max(0.0, gain_fixed_start),
                min(1.0, gain_fixed_stop),
            )
        else:
            raise RuntimeError(
                f"unsupported ro_optimize gain range mode: {gain_range_mode!r}"
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
        plan = NodeOverridePlan()
        plan.pulse_module_dependency("pi_pulse")
        plan.readout_dependency(source="readout module dependency")
        plan.generated_if(
            knobs.get("relax_delay_mode") == _RELAX_DELAY_MODE_AUTO_T1,
            "relax_delay",
            source="generation.relax.relax_delay_mode",
            reason="relax delay is generated from T1 feedback",
        )
        plan.generated_if(
            knobs.get("freq_range_mode") == _RANGE_MODE_PREVIOUS_BEST,
            "sweep.freq",
            source="generation.freq_search.freq_range_mode",
            reason="readout frequency window is generated from previous best after the first point",
            mode="after_first_point",
        )
        plan.generated_if(
            knobs.get("gain_range_mode") == _RANGE_MODE_PREVIOUS_BEST,
            "sweep.gain",
            source="generation.gain_search.gain_range_mode",
            reason="readout gain window is generated from previous best after the first point",
            mode="after_first_point",
        )
        return plan.build()

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
        knobs = env.knobs_view()
        freq_range_knob = knobs["freq_range"]
        gain_range_knob = knobs["gain_range"]

        freq_fixed_start = float(freq_range_knob.start)
        freq_fixed_stop = float(freq_range_knob.stop)
        freq_range_mode = str(knobs["freq_range_mode"])
        if freq_range_mode == _RANGE_MODE_PREVIOUS_BEST:
            freq_window_mode = str(knobs["freq_window_mode"])
            if freq_window_mode == _WINDOW_MODE_DEFAULT_SWEEP:
                freq_width = abs(freq_fixed_stop - freq_fixed_start) / 2.0
            elif freq_window_mode == _WINDOW_MODE_FIXED_HALF_WIDTH:
                freq_width = float(knobs["freq_half_width_mhz"])
            else:
                raise RuntimeError(
                    f"unsupported ro_optimize freq window mode: {freq_window_mode!r}"
                )
            freq_center = snapshot_float(
                snapshot,
                "best_ro_freq",
                _center_of(freq_range_knob),
            )
            freq_range = (freq_center - freq_width, freq_center + freq_width)
        elif freq_range_mode == _RANGE_MODE_FIXED:
            freq_range = (freq_fixed_start, freq_fixed_stop)
        else:
            raise RuntimeError(
                f"unsupported ro_optimize freq range mode: {freq_range_mode!r}"
            )

        gain_fixed_start = float(gain_range_knob.start)
        gain_fixed_stop = float(gain_range_knob.stop)
        gain_range_mode = str(knobs["gain_range_mode"])
        if gain_range_mode == _RANGE_MODE_PREVIOUS_BEST:
            gain_window_mode = str(knobs["gain_window_mode"])
            if gain_window_mode == _WINDOW_MODE_DEFAULT_SWEEP:
                gain_width = abs(gain_fixed_stop - gain_fixed_start) / 2.0
            elif gain_window_mode == _WINDOW_MODE_FIXED_HALF_WIDTH:
                gain_width = float(knobs["gain_half_width"])
            else:
                raise RuntimeError(
                    f"unsupported ro_optimize gain window mode: {gain_window_mode!r}"
                )
            gain_center = snapshot_float(
                snapshot,
                "best_ro_gain",
                _center_of(gain_range_knob),
            )
            gain_range = (
                max(0.0, gain_center - gain_width),
                min(1.0, gain_center + gain_width),
            )
        elif gain_range_mode == _RANGE_MODE_FIXED:
            gain_range = (
                max(0.0, gain_fixed_start),
                min(1.0, gain_fixed_stop),
            )
        else:
            raise RuntimeError(
                f"unsupported ro_optimize gain range mode: {gain_range_mode!r}"
            )

        t1 = snapshot_float(snapshot, "t1", float(knobs["t1_seed_us"]))
        relax_delay_mode = str(knobs["relax_delay_mode"])
        if relax_delay_mode == _RELAX_DELAY_MODE_AUTO_T1:
            relax_delay = auto_relax_delay_from_t1(
                t1,
                factor=float(knobs["relax_factor"]),
                minimum=None,
            )
        elif relax_delay_mode == _RELAX_DELAY_MODE_FIXED:
            relax_delay = float(knobs["relax_delay"])
        else:
            raise RuntimeError(
                f"unsupported ro_optimize relax_delay_mode: {relax_delay_mode!r}"
            )

        patches: dict[str, object] = {}
        patches.update(pulse_module_patches("pi_pulse", pi_pulse))
        patches.update(readout_module_patches(readout))
        if relax_delay_mode == _RELAX_DELAY_MODE_AUTO_T1:
            patches["relax_delay"] = relax_delay
        if env.flux_idx > 0 and freq_range_mode == _RANGE_MODE_PREVIOUS_BEST:
            patches["sweep.freq"] = freq_range
        if env.flux_idx > 0 and gain_range_mode == _RANGE_MODE_PREVIOUS_BEST:
            patches["sweep.gain"] = gain_range
        raw_cfg = self.point_cfg(env, patches)
        ranges = pop_sweep_ranges(raw_cfg, ("freq", "gain"), node_name=self.name)
        raw_cfg["freq_range"] = ranges["freq"]
        raw_cfg["gain_range"] = ranges["gain"]
        return ml.make_cfg(raw_cfg, RoOptimizeCfgTemplate)

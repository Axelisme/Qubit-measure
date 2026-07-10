"""qubit_freq — two-tone acquire around a predicted drive center.

The Builder owns cfg lowering and feedback policy; the short-lived Node performs
one flux point's real acquire, fit, Result fill, and Patch emission. See
``CONTEXT.md`` for the Builder/Node/orchestrator boundary.

- ``predict_freq`` — required; provided by the predictor Service (a Builder
  whose Node computes it), resolved latest-available like any dependency.
- ``fit_kappa`` — raw fit FWHM reported only when the stricter linewidth gate
  trusts it for downstream consumers.
- ``qfw_factor`` — adaptive drive feedback, conditionally reported as
  ``fit_kappa / gain`` and read back smoothed with the notebook's step-weighted
  rule to choose the next drive gain.
- ``readout`` — optional module, Node-produced (ro_optimize) → ml preset →
  default.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.gui.app.autofluxdep.cfg import OverridePlan
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema, sweepcfg_to_axis
from zcu_tools.gui.app.autofluxdep.experiments._support.acquire import (
    DEFAULT_ACQUIRE_RETRY,
    SnrProbe,
    acquire_retry,
    acquire_to_complex,
    axis_to_sweep,
    build_stop_condition,
    is_good_fit,
    make_signal_update,
    schedule_completed,
    setup_flux_point,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.dependency_defaults import (
    missing_info_value,
    missing_module_value,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.module_aliases import (
    READOUT_LIBRARY_ALIASES,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.plotters import title_with_snr
from zcu_tools.gui.app.autofluxdep.experiments._support.result import QubitFreqResult
from zcu_tools.gui.app.autofluxdep.experiments._support.utils import (
    NodeOverridePlan,
    NodeSchemaBuilder,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.utils.override_plan import (
    readout_module_patches,
)
from zcu_tools.gui.app.autofluxdep.feedback import FeedbackSlotDecl, ScalarEstimator
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.tools import Predictor
from zcu_tools.gui.cfg import CenteredSweepValue
from zcu_tools.gui.session.types import ExpContext
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    ResetCfg,
    sweep2param,
)
from zcu_tools.simulate.fluxonium.physical_fit import (
    FluxoniumLocalFitResult,
    fit_local_fluxonium_model,
)
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import rotate2real

logger = logging.getLogger(__name__)

_DRIVE_GAIN_CAP = 1.0
_DRIVE_GAIN_MODE_ADAPTIVE = "adaptive"
_DRIVE_GAIN_MODE_FIXED = "fixed"
_PREDICT_FREQ_CORRECTION_SLOT = FeedbackSlotDecl(
    key="predict_freq_correction",
    kind="estimator",
    prefix="pred_freq_correction",
    default_strategy="idw",
    default_idw_k=10,
    default_idw_epsilon=1e-4,
    default_decay_points=4.0,
)


class QubitFreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class QubitFreqCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The explicit qubit-frequency cfg lowered from the active context.

    ``ProgramV2Cfg`` supplies the program runtime fields, this node owns its
    module shape, and ``ExpCfgModel`` supplies the experiment fields. The flux
    ``dev`` entry and ``detune`` sweep are merged in by ``produce`` (the sweep
    recenters on the predicted frequency and the device carries this point's
    flux value), mirroring the lower-layer ``experiment/v2/autofluxdep``
    ``QubitFreqCfgTemplate``.
    """

    modules: QubitFreqModuleCfg


def _drive_gain_from_qfw_factor(
    qfw_factor: Any | None,
    initial_gain: float,
    *,
    target_kappa: float,
    drive_gain_cap: float,
) -> float:
    if qfw_factor is None:
        return float(initial_gain)
    factor = float(qfw_factor)
    if factor <= 0.0:
        raise RuntimeError(
            "qubit_freq.make_cfg needs positive qfw_factor to derive drive gain"
        )
    return min(float(drive_gain_cap), float(target_kappa) / factor)


def _predict_freq_correction(env: RunEnv) -> float:
    if env.feedback is None:
        return 0.0
    estimator = env.feedback.estimator(_PREDICT_FREQ_CORRECTION_SLOT.key)
    if estimator is None:
        return 0.0
    estimate = estimator.estimate(env.flux)
    if estimate is None:
        return 0.0
    return float(estimate.confidence * estimate.value)


def _observe_predict_freq_residual(
    env: RunEnv, measured_freq: float, base_after_calibration: float
) -> None:
    if env.feedback is None:
        return
    estimator = env.feedback.estimator(_PREDICT_FREQ_CORRECTION_SLOT.key)
    if estimator is None:
        return
    estimator.observe(env.flux, float(measured_freq) - float(base_after_calibration))


def _signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """PCA-rotate to the real axis and normalise to [0, 1] (a dip near 0)."""
    real = rotate2real(np.asarray(signals, dtype=np.complex128)).real
    lo, hi = float(real.min()), float(real.max())
    real = (real - lo) / (hi - lo + 1e-12)
    # orient so the resonance is a dip (start/end high, centre low)
    if real[0] + real[-1] < real[len(real) // 2]:
        real = 1.0 - real
    return real


def _is_trusted_frequency_fit(
    real: NDArray[np.float64], fit_curve: NDArray[np.float64]
) -> bool:
    return is_good_fit(real, fit_curve, threshold=0.2)


def _is_trusted_linewidth_fit(
    fwhm: float,
    detunes: NDArray[np.float64],
    real: NDArray[np.float64],
    fit_curve: NDArray[np.float64],
) -> bool:
    width = float(fwhm)
    if not np.isfinite(width) or width <= 0.0:
        return False

    axis = np.asarray(detunes, dtype=np.float64)
    if axis.size < 2 or not np.all(np.isfinite(axis)):
        return False
    span = float(np.ptp(axis))
    if not np.isfinite(span) or span <= 0.0 or width > span:
        return False

    return is_good_fit(real, fit_curve, threshold=0.1)


class QubitFreqNode(Node):
    """One flux point's qubit_freq execution, environment curried in by build_node.

    Sets this point's flux on the picked flux device, recenters the detune sweep
    on the snapshot's ``predict_freq``, runs a real Schedule-backed
    ``ModularProgramV2`` acquire against the connected board (the flux-aware MockSoc
    offline, or real hardware),
    fits the dip with ``fit_qubit_freq``, fills the sweep Result's ``flux_idx`` row
    in place, fires the ``round_hook`` so the main thread redraws, and returns the
    raw fit Patch.
    """

    def __init__(self, env: RunEnv, builder: QubitFreqBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        base_pred_qf = float(snapshot["predict_freq"])

        result: QubitFreqResult = env.result
        idx = env.flux_idx
        detunes = result.detune  # MHz, relative to the drive centre

        # Build the run cfg from the active context (Fast Fail if unconfigured: a
        # real acquire needs a concrete readout + drive pulse). The drive centre is
        # the predicted qubit freq (make_cfg sets qub_pulse.freq = predict_freq).
        cfg = self._builder.make_cfg(env, snapshot)
        center = float(cfg.modules.qub_pulse.freq)
        pred_qf = center
        freqs = center + detunes  # absolute frequency axis (for the fit + plot)

        # Point the flux device at this sweep point and push it to hardware (mock:
        # writes the FakeDevice value → SimEngine reads it live). Fast Fail if no
        # flux source is picked — a real flux sweep must drive a device.
        setup_flux_point(cfg, env, "qubit_freq")

        # Recenter the detune sweep on the predicted freq (mirrors the lower layer:
        # qub_pulse.freq + detune_param), so the FPGA sweeps freq across the detune
        # window around the drive centre.
        detune_sweep = axis_to_sweep(detunes)
        detune_param = sweep2param("detune", detune_sweep)
        cfg.modules.qub_pulse.set_param(
            "freq", cfg.modules.qub_pulse.freq + detune_param
        )

        result.flux[idx] = env.flux
        result.predict_freq[idx] = pred_qf

        # Real multi-round acquire. round_hook fires per round with the running
        # average; we rotate it to real, overwrite the Result row, and notify so the
        # liveplot settles round by round. The SNR probe + stop poll are evaluated
        # as a completed-round stop condition.
        probe = SnrProbe()
        signal_buffer = SignalBuffer(
            result.signal[idx].shape,
            dtype=np.complex128,
            on_update=make_signal_update(
                result,
                idx,
                _signal2real,
                env.round_hook,
                probe=probe,
            ),
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
                    Pulse("init_pulse", cfg.modules.init_pulse, tag="init_pulse"),
                    Pulse("qubit_pulse", cfg.modules.qub_pulse, tag="qub_pulse"),
                    Readout("readout", cfg.modules.readout),
                ]
            ).declare_sweep("detune", detune_sweep)
            signal = builder.build_and_acquire(
                raw2signal_fn=acquire_to_complex,
                retry=acquire_retry(env),
                progress=False,
                progress_label=f"{env.node_name or 'qubit_freq'} flux {idx + 1} rounds",
                progress_leave=False,
                stop_condition=build_stop_condition(env, probe),
            )
            outcome = sched.outcome

        if not schedule_completed(outcome, "qubit_freq"):
            return Patch()

        real = _signal2real(np.asarray(signal, dtype=np.complex128))

        # fit the fully-averaged signal
        freq, _, fwhm, _, fit_curve, _ = fit_qubit_freq(freqs, real)

        # fit-quality gate (the runner module's mean_err vs ptp): a poor fit is
        # discarded — it does NOT enter the Result, does NOT update predictor or
        # feedback state, and is omitted from the Patch so downstream falls back
        # to the latest good value.
        if not _is_trusted_frequency_fit(real, fit_curve):
            logger.debug(
                "qubit_freq fit @flux%d: poor fit (SNR-trough?) — "
                "discarded, no calibrate, no qubit_freq produced",
                idx,
            )
            on_fit_failed(
                env,
                snapshot_predict_freq=base_pred_qf,
                estimator_key=_PREDICT_FREQ_CORRECTION_SLOT.key,
            )
            if env.round_hook is not None:
                env.round_hook(idx)  # raw row already shown; fit fields stay nan
            return Patch()  # partial: omit qubit_freq → downstream latest-available

        # good fit: fill the Result's fit fields + update the configured bias path
        result.fit_freq[idx] = float(freq)
        np.copyto(result.fit_curve[idx], np.asarray(fit_curve, dtype=np.float64))
        if env.round_hook is not None:
            env.round_hook(idx)

        # The physical/base predictor stays immutable during the run. Physical
        # recovery may install a run-local overlay; generic feedback carries the
        # remaining residual correction.
        knobs = env.knobs_view()
        recovery_reseeded = on_fit_succeeded(
            env,
            float(freq),
            snapshot_predict_freq=base_pred_qf,
            estimator_key=_PREDICT_FREQ_CORRECTION_SLOT.key,
        )
        base_for_residual = physical_prediction_for_make_cfg(env, base_pred_qf, knobs)
        if not recovery_reseeded:
            _observe_predict_freq_residual(env, float(freq), base_for_residual)

        logger.debug(
            "qubit_freq fit @flux%d: freq=%.3f (predict=%.3f, detune=%+.3f) kappa=%.3f"
            " → updated prediction feedback",
            idx,
            float(freq),
            pred_qf,
            float(freq) - pred_qf,
            float(fwhm),
        )

        patch = Patch()
        patch.set("qubit_freq", float(freq))
        patch.set("fit_detune", float(freq) - pred_qf)
        if not _is_trusted_linewidth_fit(float(fwhm), detunes, real, fit_curve):
            logger.debug(
                "qubit_freq fit @flux%d: linewidth feedback rejected "
                "(kappa=%.3f); frequency fit kept",
                idx,
                float(fwhm),
            )
            return patch

        patch.set("fit_kappa", float(fwhm))
        drive_gain = float(cfg.modules.qub_pulse.gain)
        if drive_gain <= 0.0:
            raise RuntimeError(
                "qubit_freq produced fit_kappa with non-positive drive gain"
            )
        patch.set("qfw_factor", float(fwhm) / drive_gain)
        return patch


class QubitFreqPlotter:
    """qubit_freq's two-panel liveplot, aligned with the runner module.

    Built once at Run start with a bare matplotlib ``Figure``; reuses
    ``zcu_tools.liveplot`` (LivePlot1D / LivePlot2DwithLine) embedded into the
    Figure's axes via ``existed_axes`` (the liveplot fig is None then — the host
    refreshes; see ``zcu_tools.liveplot.segments.base``). ``update(result, idx)``
    on the main thread after each row notification feeds:

    - ``fit_freq`` (LivePlot1D): flux value → fitted absolute qubit frequency.
    - ``detune`` (LivePlot2DwithLine): the flux × detune signal colormap plus the
      latest few flux rows as 1-D traces; a red dashed line marks the current
      fit_detune (matching the runner's ``freq_line``).
    """

    def __init__(self, figure: Any) -> None:
        from zcu_tools.liveplot import LivePlot1D, LivePlot2DwithLine

        self._fig = figure
        ax_fit = figure.add_subplot(2, 1, 1)
        ax_2d = figure.add_subplot(2, 2, 3)
        ax_line = figure.add_subplot(2, 2, 4)
        self._detune_title = "qubit_freq (detune)"
        self._freq_line = ax_line.axvline(np.nan, color="red", linestyle="--")
        self._fit = LivePlot1D(
            "Flux device value",
            "Frequency (MHz)",
            existed_axes=[[ax_fit]],
            segment_kwargs=dict(title="qubit_freq (fit_freq)"),
        )
        self._detune = LivePlot2DwithLine(
            "Flux device value",
            "Detune (MHz)",
            line_axis=1,
            num_lines=3,
            title=self._detune_title,
            existed_axes=[[ax_2d, ax_line]],
        )
        self._fit.__enter__()
        self._detune.__enter__()

    def update(self, result: QubitFreqResult, idx: int) -> None:
        self._fit.update(result.flux, result.fit_freq, refresh=False)
        self._detune.update(
            result.flux,
            result.detune,
            result.signal,
            title=title_with_snr(self._detune_title, result, idx),
            refresh=False,
        )
        # mark the current fit as a detune offset (freq - predict_freq)
        offset = result.fit_freq - result.predict_freq
        valid = offset[~np.isnan(offset)]
        self._freq_line.set_xdata([valid[-1] if valid.size else np.nan])
        self._fig.canvas.draw_idle()


class QubitFreqBuilder(Builder):
    """The qubit_freq provider — stateless; builds Result / Plotter / Nodes.

    Reports trusted raw fit results. Frequency keys (qubit_freq, fit_detune) use
    the basic fit gate; linewidth feedback keys (fit_kappa, qfw_factor) use a
    stricter gate. The provider never smooths its own output; the orchestrator
    projects the smoothed qfw_factor into the next point.
    """

    name = "qubit_freq"
    provides = ("qubit_freq", "fit_detune", "fit_kappa", "qfw_factor")
    # predict_freq is required, supplied by the predictor Service (a Builder
    # whose Node computes it). With latest-available resolution a consumer
    # ordered before the predictor just reads the previous point's value.
    requires = (Dependency("predict_freq"),)
    optional = (
        # consumer-declared smoothing: read qfw_factor *smoothed* (same key). The
        # first point uses the schema drive gain; later points follow the notebook's
        # step-weighted linewidth/gain feedback. The orchestrator builds the
        # SmoothingService from this declaration alone.
        Dependency("qfw_factor", smooth="step_weighted", default=missing_info_value),
    )
    # the readout module: Node-produced → calibrated ml preset → default.
    optional_modules = (
        ModuleDep(
            "readout", default=missing_module_value, aliases=READOUT_LIBRARY_ALIASES
        ),
    )
    feedback_slots = (_PREDICT_FREQ_CORRECTION_SLOT,)

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Default cfg plus autofluxdep generation controls."""
        qub_ch = 0
        if isinstance(ctx, ExpContext):
            value = ctx.md.get("qub_ch")
            if isinstance(value, int) and not isinstance(value, bool):
                qub_ch = value

        return (
            NodeSchemaBuilder(ctx, label="Qubit Frequency")
            .pulse(
                "qub_pulse",
                "modules.qub_pulse",
                label="Probe Pulse",
                library_keys=("qub_probe",),
                overrides={
                    "ch": qub_ch,
                    "freq": 0.0,
                    "gain": 0.1,
                    "waveform.length": 5.0,
                },
            )
            .pulse_readout(
                "readout",
                "modules.readout",
                label="Readout",
                library_keys=READOUT_LIBRARY_ALIASES,
            )
            .float("relax_delay", "relax_delay", label="Relax delay (us)", default=1.0)
            .int("reps", "reps", label="Reps", default=1000)
            .int("rounds", "rounds", label="Rounds", default=10)
            .centered_sweep(
                "detune_sweep",
                "sweep.freq",
                label="Freq (MHz)",
                default=CenteredSweepValue(center=0.0, span=100.0, expts=201),
                center_editable=False,
                center_badge="generated",
                locked_center=0.0,
                center_tooltip=(
                    "Detune-window center is fixed to the generated drive center; "
                    "edit span/expts to control the search window."
                ),
                tooltip="Detune search window around the generated qubit drive center.",
            )
            .knob("qub_ch", "modules.qub_pulse.ch")
            .knob("qub_nqz", "modules.qub_pulse.nqz")
            .knob("qub_gain", "modules.qub_pulse.gain")
            .knob("qub_length", "modules.qub_pulse.waveform.length")
            .acquisition(
                retry=DEFAULT_ACQUIRE_RETRY,
                early_stop_snr=50.0,
            )
            .choice(
                "physical_recovery_mode",
                "generation.freq_recovery.physical_recovery_mode",
                label="mode",
                choices=(
                    PHYSICAL_RECOVERY_MODE_OFF,
                    PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT,
                ),
                default=PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT,
                tooltip="Run-local physical-model recovery after trusted frequency fits fail.",
            )
            .choice(
                "drive_gain_mode",
                "generation.drive_gain.drive_gain_mode",
                label="mode",
                choices=(_DRIVE_GAIN_MODE_ADAPTIVE, _DRIVE_GAIN_MODE_FIXED),
                default=_DRIVE_GAIN_MODE_ADAPTIVE,
                tooltip="Adaptive uses linewidth feedback; fixed keeps Default cfg gain.",
            )
            .float(
                "target_kappa",
                "generation.drive_gain.target_kappa",
                label="target_kappa",
                default=6.5,
                tooltip="Target linewidth in MHz for adaptive drive gain.",
            )
            .choice_fields(
                "generation.drive_gain",
                "drive_gain_mode",
                {
                    _DRIVE_GAIN_MODE_FIXED: (),
                    _DRIVE_GAIN_MODE_ADAPTIVE: ("target_kappa",),
                },
            )
            .feedback_slot(
                _PREDICT_FREQ_CORRECTION_SLOT,
                group="predictor_correction",
            )
            .build()
        )

    def make_init_result(
        self, schema: NodeCfgSchema, flux: Any, md: Any = None
    ) -> QubitFreqResult:
        knobs = schema.lower(None, md=md)
        detune = sweepcfg_to_axis(knobs["detune_sweep"])
        return QubitFreqResult.allocate(flux, detune)

    def make_plotter(self, figure: Any) -> QubitFreqPlotter:
        return QubitFreqPlotter(figure)

    def build_node(self, env: RunEnv) -> QubitFreqNode:
        return QubitFreqNode(env, self)

    def override_plan(self, schema: NodeCfgSchema) -> OverridePlan:
        knobs = schema.read_knobs()
        plan = NodeOverridePlan()
        plan.generated_if(
            True,
            "modules.qub_pulse.freq",
            source="predict_freq",
            reason="qubit drive frequency is generated from predictor feedback",
        )
        plan.generated_if(
            knobs.get("drive_gain_mode") == _DRIVE_GAIN_MODE_ADAPTIVE,
            "modules.qub_pulse.gain",
            source="generation.drive_gain.drive_gain_mode",
            reason="adaptive drive gain is generated from linewidth feedback after the initial point",
            mode="after_first_point",
        )
        plan.readout_dependency(source="readout module dependency")
        return plan.build()

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> QubitFreqCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the notebook's qubit_freq ``cfg_maker`` (D1: runs in ``produce``,
        where the snapshot is available): the drive pulse frequency is the
        predicted qubit freq, the readout is the latest-available readout module,
        and the pulse waveform / channel / gain / nqz come from the node's params
        (the "設定頭"). The flux ``dev`` entry and the ``detune`` sweep are NOT here
        — ``produce`` merges them (the dev with this point's flux value, the sweep
        recentred on the predicted freq).

        Raises if the readout module is unavailable or the drive params are unset
        — a real run needs a concrete drive pulse (Fast Fail).
        """
        ml = env.ml
        if ml is None:
            raise RuntimeError("qubit_freq.make_cfg needs an active ModuleLibrary")
        readout = snapshot.module("readout")
        if readout is None:
            raise RuntimeError(
                "qubit_freq.make_cfg needs a readout module (none produced or preset)"
            )
        knobs = env.knobs_view()
        base_predict_freq = physical_prediction_for_make_cfg(
            env,
            float(snapshot["predict_freq"]),
            knobs,
        )
        predict_freq = base_predict_freq + _predict_freq_correction(env)
        patches: dict[str, object] = {
            "modules.qub_pulse.freq": predict_freq,
        }
        drive_gain_mode = str(knobs["drive_gain_mode"])
        if drive_gain_mode == _DRIVE_GAIN_MODE_ADAPTIVE:
            if env.flux_idx > 0:
                patches["modules.qub_pulse.gain"] = _drive_gain_from_qfw_factor(
                    snapshot.get("qfw_factor"),
                    float(knobs["qub_gain"]),
                    target_kappa=float(knobs["target_kappa"]),
                    drive_gain_cap=_DRIVE_GAIN_CAP,
                )
        elif drive_gain_mode != _DRIVE_GAIN_MODE_FIXED:
            raise RuntimeError(
                f"unsupported qubit_freq drive_gain_mode: {drive_gain_mode!r}"
            )
        patches.update(readout_module_patches(readout))
        raw_cfg = self.point_cfg(env, patches)
        raw_cfg.pop("sweep", None)
        return ml.make_cfg(raw_cfg, QubitFreqCfgTemplate)


# qubit_freq-only fail-triggered physical recovery policy
PHYSICAL_RECOVERY_MODE_OFF = "off"
PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT = "fail_triggered_fit"
PhysicalRecoveryMode = Literal["off", "fail_triggered_fit"]

DEFAULT_PHYSICAL_RECOVERY_MIN_POINTS = 10
DEFAULT_PHYSICAL_RECOVERY_MAX_POINTS = 30
DEFAULT_PHYSICAL_RECOVERY_MAX_CENTER_SHIFT_MHZ = 150.0
DEFAULT_PHYSICAL_RECOVERY_MAX_RMS_MHZ = 50.0


@dataclass(frozen=True)
class TrustedFrequencyPoint:
    flux: float
    frequency_mhz: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "flux", _finite("trusted flux", self.flux))
        object.__setattr__(
            self,
            "frequency_mhz",
            _finite("trusted frequency", self.frequency_mhz),
        )


@dataclass(frozen=True)
class PhysicalRecoveryConfig:
    mode: PhysicalRecoveryMode
    min_points: int
    max_points: int
    max_center_shift_mhz: float
    max_rms_mhz: float

    @property
    def enabled(self) -> bool:
        return self.mode == PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT


@dataclass(frozen=True)
class PhysicalRecoveryAttempt:
    trigger: str
    accepted: bool
    reason: str
    n_points: int
    base_rms_mhz: float
    fitted_rms_mhz: float
    center_shift_mhz: float


@dataclass
class QubitFreqRecoveryState:
    history: list[TrustedFrequencyPoint] = field(default_factory=list)
    fail_streak: int = 0
    overlay: Predictor | None = None
    last_attempt: PhysicalRecoveryAttempt | None = None


def recovery_config_from_knobs(knobs: Mapping[str, Any]) -> PhysicalRecoveryConfig:
    mode = str(knobs["physical_recovery_mode"])
    if mode not in (
        PHYSICAL_RECOVERY_MODE_OFF,
        PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT,
    ):
        raise RuntimeError(f"unsupported qubit_freq physical_recovery_mode: {mode!r}")

    return PhysicalRecoveryConfig(
        mode=cast(PhysicalRecoveryMode, mode),
        min_points=DEFAULT_PHYSICAL_RECOVERY_MIN_POINTS,
        max_points=DEFAULT_PHYSICAL_RECOVERY_MAX_POINTS,
        max_center_shift_mhz=DEFAULT_PHYSICAL_RECOVERY_MAX_CENTER_SHIFT_MHZ,
        max_rms_mhz=DEFAULT_PHYSICAL_RECOVERY_MAX_RMS_MHZ,
    )


def physical_prediction_for_make_cfg(
    env: RunEnv,
    snapshot_predict_freq: float,
    knobs: Mapping[str, Any],
) -> float:
    """Return active physical prediction before generic residual correction."""
    cfg = recovery_config_from_knobs(knobs)
    if not cfg.enabled or env.tools is None:
        return float(snapshot_predict_freq)
    state = env.tools.peek_recovery_state(_placement_key(env), QubitFreqRecoveryState)
    if state is None or state.overlay is None:
        return float(snapshot_predict_freq)
    return float(state.overlay.predict_freq(env.flux))


def on_fit_failed(
    env: RunEnv,
    *,
    snapshot_predict_freq: float,
    estimator_key: str,
) -> None:
    knobs = env.knobs_view()
    cfg = recovery_config_from_knobs(knobs)
    if not cfg.enabled or env.tools is None:
        return
    state = env.tools.recovery_state(_placement_key(env), QubitFreqRecoveryState)
    state.fail_streak += 1
    if state.fail_streak == 1:
        _attempt_recovery_fit(
            env,
            cfg,
            state,
            trigger="first_fail",
            snapshot_predict_freq=snapshot_predict_freq,
            estimator_key=estimator_key,
        )


def on_fit_succeeded(
    env: RunEnv,
    measured_freq_mhz: float,
    *,
    snapshot_predict_freq: float,
    estimator_key: str,
) -> bool:
    knobs = env.knobs_view()
    cfg = recovery_config_from_knobs(knobs)
    if not cfg.enabled or env.tools is None:
        return False
    state = env.tools.recovery_state(_placement_key(env), QubitFreqRecoveryState)
    state.history.append(TrustedFrequencyPoint(env.flux, measured_freq_mhz))
    had_fail = state.fail_streak > 0
    reseeded = False
    if had_fail:
        reseeded = _attempt_recovery_fit(
            env,
            cfg,
            state,
            trigger="first_success_after_fail",
            snapshot_predict_freq=snapshot_predict_freq,
            estimator_key=estimator_key,
        )
    state.fail_streak = 0
    return reseeded


def select_fit_points(
    history: Iterable[TrustedFrequencyPoint],
    *,
    min_points: int = DEFAULT_PHYSICAL_RECOVERY_MIN_POINTS,
    max_points: int = DEFAULT_PHYSICAL_RECOVERY_MAX_POINTS,
) -> tuple[TrustedFrequencyPoint, ...]:
    """Select a deterministic, flux-spread subset for local physical fitting."""
    if not (
        DEFAULT_PHYSICAL_RECOVERY_MIN_POINTS
        <= min_points
        <= max_points
        <= DEFAULT_PHYSICAL_RECOVERY_MAX_POINTS
    ):
        raise RuntimeError("fit point bounds must satisfy 10 <= min <= max <= 30")

    latest_by_flux: dict[float, TrustedFrequencyPoint] = {}
    for point in history:
        trusted = TrustedFrequencyPoint(point.flux, point.frequency_mhz)
        latest_by_flux[trusted.flux] = trusted

    points = tuple(sorted(latest_by_flux.values(), key=lambda point: point.flux))
    if len(points) < min_points:
        return ()
    if len(points) <= max_points:
        return points

    first, last = points[0], points[-1]
    span = last.flux - first.flux
    if span <= 0.0:
        return ()

    low = 0.0
    high = span / float(max_points - 1)
    while len(_greedy_spacing(points, high)) > max_points and high < span:
        high *= 2.0

    best = _greedy_spacing(points, high)
    for _ in range(64):
        mid = (low + high) / 2.0
        candidate = _greedy_spacing(points, mid)
        if len(candidate) > max_points:
            low = mid
        else:
            best = candidate
            high = mid

    selected = _prefer_last_endpoint(best, points[-1], high)
    if len(selected) > max_points:
        selected = _thin_evenly(selected, max_points)
    if len(selected) < min_points:
        return ()
    return tuple(selected)


def _attempt_recovery_fit(
    env: RunEnv,
    cfg: PhysicalRecoveryConfig,
    state: QubitFreqRecoveryState,
    *,
    trigger: str,
    snapshot_predict_freq: float,
    estimator_key: str,
) -> bool:
    selected = select_fit_points(
        state.history,
        min_points=cfg.min_points,
        max_points=cfg.max_points,
    )
    if not selected:
        state.last_attempt = _attempt(
            trigger,
            False,
            "not enough trusted history",
            0,
            math.nan,
            math.nan,
            math.nan,
        )
        return False

    predictor = state.overlay
    if predictor is None and env.tools is not None:
        predictor = env.tools.predictor
    if predictor is None:
        state.last_attempt = _attempt(
            trigger,
            False,
            "no physical predictor",
            len(selected),
            math.nan,
            math.nan,
            math.nan,
        )
        return False
    if not predictor.supports_physical_recovery():
        state.last_attempt = _attempt(
            trigger,
            False,
            "predictor does not support physical recovery",
            len(selected),
            math.nan,
            math.nan,
            math.nan,
        )
        return False

    base = predictor.physical_snapshot()
    fit = fit_local_fluxonium_model(
        base,
        ((point.flux, point.frequency_mhz) for point in selected),
    )
    candidate: Predictor | None = None
    if fit.accepted and fit.fitted is not None:
        try:
            candidate = predictor.overlay_physical(fit.fitted)
        except Exception as exc:
            raise RuntimeError("qubit_freq physical recovery overlay failed") from exc

    estimator = None
    if env.feedback is not None:
        estimator = env.feedback.estimator(estimator_key)
    attempt = _accept_candidate(
        env,
        cfg,
        trigger,
        selected,
        predictor,
        candidate,
        fit,
        snapshot_predict_freq=snapshot_predict_freq,
        estimator=estimator,
    )
    state.last_attempt = attempt
    if attempt.accepted and candidate is not None:
        state.overlay = candidate
        logger.info(
            "qubit_freq physical recovery accepted @flux%d via %s: "
            "n=%d rms %.3f->%.3f shift %.3f MHz",
            env.flux_idx,
            trigger,
            attempt.n_points,
            attempt.base_rms_mhz,
            attempt.fitted_rms_mhz,
            attempt.center_shift_mhz,
        )
    else:
        logger.debug(
            "qubit_freq physical recovery rejected @flux%d via %s: %s",
            env.flux_idx,
            trigger,
            attempt.reason,
        )
    return attempt.accepted and candidate is not None


def _accept_candidate(
    env: RunEnv,
    cfg: PhysicalRecoveryConfig,
    trigger: str,
    selected: tuple[TrustedFrequencyPoint, ...],
    base_predictor: Predictor,
    candidate: Predictor | None,
    fit: FluxoniumLocalFitResult,
    *,
    snapshot_predict_freq: float,
    estimator: ScalarEstimator | None,
) -> PhysicalRecoveryAttempt:
    if len(selected) < cfg.min_points or len(selected) > cfg.max_points:
        return _attempt(
            trigger,
            False,
            "selected point count outside configured bounds",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )
    if not fit.accepted or fit.fitted is None:
        return _attempt(
            trigger,
            False,
            fit.reason,
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )
    if candidate is None:
        return _attempt(
            trigger,
            False,
            "overlay predictor unavailable",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )
    if not (math.isfinite(fit.base_rms_mhz) and math.isfinite(fit.fitted_rms_mhz)):
        return _attempt(
            trigger,
            False,
            "fit RMS is not finite",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )
    if fit.fitted_rms_mhz > fit.base_rms_mhz:
        return _attempt(
            trigger,
            False,
            "fit RMS is worse than active base",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )
    if fit.fitted_rms_mhz > cfg.max_rms_mhz:
        return _attempt(
            trigger,
            False,
            "fit RMS exceeds physical_recovery_max_rms_mhz",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            math.nan,
        )

    try:
        base_center = float(base_predictor.predict_freq(env.flux))
    except Exception:
        base_center = float(snapshot_predict_freq)
    try:
        candidate_center = float(candidate.predict_freq(env.flux))
    except Exception as exc:
        raise RuntimeError(
            "qubit_freq physical recovery candidate prediction failed"
        ) from exc
    center_shift = abs(candidate_center - base_center)
    if not math.isfinite(center_shift):
        return _attempt(
            trigger,
            False,
            "candidate center shift is not finite",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            center_shift,
        )
    if center_shift > cfg.max_center_shift_mhz:
        return _attempt(
            trigger,
            False,
            "candidate center shift exceeds physical_recovery_max_center_shift_mhz",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            center_shift,
        )
    if estimator is None:
        return _attempt(
            trigger,
            True,
            "accepted without correction reseed",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            center_shift,
        )

    residuals = tuple(
        (point.flux, point.frequency_mhz - float(candidate.predict_freq(point.flux)))
        for point in selected
    )
    if any(
        not (math.isfinite(flux) and math.isfinite(residual))
        for flux, residual in residuals
    ):
        return _attempt(
            trigger,
            False,
            "reseed residual is not finite",
            len(selected),
            fit.base_rms_mhz,
            fit.fitted_rms_mhz,
            center_shift,
        )
    estimator.replace_observations(residuals)
    return _attempt(
        trigger,
        True,
        "accepted",
        len(selected),
        fit.base_rms_mhz,
        fit.fitted_rms_mhz,
        center_shift,
    )


def _placement_key(env: RunEnv) -> str:
    return env.node_name or "qubit_freq"


def _greedy_spacing(
    points: tuple[TrustedFrequencyPoint, ...],
    min_spacing: float,
) -> tuple[TrustedFrequencyPoint, ...]:
    selected: list[TrustedFrequencyPoint] = []
    last_flux = -math.inf
    tolerance = max(abs(min_spacing), 1.0) * 1e-12
    for point in points:
        if not selected or point.flux - last_flux >= min_spacing - tolerance:
            selected.append(point)
            last_flux = point.flux
    return tuple(selected)


def _prefer_last_endpoint(
    selected: tuple[TrustedFrequencyPoint, ...],
    last_point: TrustedFrequencyPoint,
    min_spacing: float,
) -> tuple[TrustedFrequencyPoint, ...]:
    if not selected or selected[-1] == last_point:
        return selected
    if len(selected) == 1:
        return (last_point,)
    tolerance = max(abs(min_spacing), 1.0) * 1e-12
    if last_point.flux - selected[-2].flux >= min_spacing - tolerance:
        return (*selected[:-1], last_point)
    return selected


def _thin_evenly(
    points: tuple[TrustedFrequencyPoint, ...],
    max_points: int,
) -> tuple[TrustedFrequencyPoint, ...]:
    if len(points) <= max_points:
        return points
    raw_indices = np.linspace(0, len(points) - 1, max_points)
    indices: list[int] = []
    for raw in raw_indices:
        index = int(round(float(raw)))
        if indices and index <= indices[-1]:
            index = indices[-1] + 1
        indices.append(min(index, len(points) - 1))
    indices[-1] = len(points) - 1
    return tuple(points[index] for index in indices)


def _attempt(
    trigger: str,
    accepted: bool,
    reason: str,
    n_points: int,
    base_rms_mhz: float,
    fitted_rms_mhz: float,
    center_shift_mhz: float,
) -> PhysicalRecoveryAttempt:
    return PhysicalRecoveryAttempt(
        trigger=trigger,
        accepted=accepted,
        reason=reason,
        n_points=int(n_points),
        base_rms_mhz=float(base_rms_mhz),
        fitted_rms_mhz=float(fitted_rms_mhz),
        center_shift_mhz=float(center_shift_mhz),
    )


def _finite(name: str, value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"qubit_freq {name} must be finite")
    return out


EXPERIMENT = QubitFreqBuilder()

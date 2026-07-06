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
from collections.abc import MutableMapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2_gui.adapters.twotone.freq import FreqAdapter
from zcu_tools.gui.app.autofluxdep.cfg import (
    FloatSpec,
    IntSpec,
    OverridePath,
    OverridePlan,
    SweepValue,
    str_choice_spec,
)
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema, sweepcfg_to_axis
from zcu_tools.gui.app.autofluxdep.feedback import (
    FeedbackSlotDecl,
    feedback_generation_choice,
    feedback_generation_fields,
)
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    SnrProbe,
    acquire_retry_generation_field,
    acquire_to_complex,
    axis_to_sweep,
    build_stop_checkers,
    is_good_fit,
    make_signal_update,
    require_flux_device,
    run_schedule_acquire,
    set_flux_by_name,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.defaults import (
    PULSE_READOUT_REF_LABELS,
    adapter_node_schema,
    ctx_md_float,
    generation_choice,
    logical_generation_field,
    readout_module_override_paths,
    readout_module_patches,
)
from zcu_tools.gui.app.autofluxdep.nodes.dependency_defaults import (
    missing_module_value,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import READOUT_LIBRARY_ALIASES
from zcu_tools.gui.app.autofluxdep.nodes.plotters import title_with_snr
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq_recovery import (
    DEFAULT_PHYSICAL_RECOVERY_MAX_CENTER_SHIFT_MHZ,
    DEFAULT_PHYSICAL_RECOVERY_MAX_POINTS,
    DEFAULT_PHYSICAL_RECOVERY_MAX_RMS_MHZ,
    DEFAULT_PHYSICAL_RECOVERY_MIN_POINTS,
    PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT,
    PHYSICAL_RECOVERY_MODE_OFF,
    on_fit_failed,
    on_fit_succeeded,
    physical_prediction_for_make_cfg,
    physical_prediction_for_residual,
)
from zcu_tools.gui.app.autofluxdep.nodes.result import QubitFreqResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    Readout,
    TwoToneCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import rotate2real

logger = logging.getLogger(__name__)

_DEFAULT_EARLYSTOP_SNR = 50.0
_DEFAULT_DETUNE_SWEEP = SweepValue(start=-50.0, stop=50.0, expts=201)
_QFW_TARGET_KAPPA = 6.5
_DRIVE_GAIN_CAP = 1.0
_FREQ_FIT_RESIDUAL_RATIO = 0.2
_LINEWIDTH_FIT_RESIDUAL_RATIO = 0.1
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


class QubitFreqCfgTemplate(TwoToneCfg, ExpCfgModel):
    """The base two-tone cfg qubit_freq lowers a context into.

    Just ``TwoToneCfg`` (reps/rounds/relax + qub_pulse/readout modules) + the
    ``ExpCfgModel`` device/save fields — the flux ``dev`` entry and the ``detune``
    sweep are merged in by ``produce`` (the sweep recenters on the predicted freq,
    and the dev carries this flux point's value), mirroring the lower-layer
    ``experiment/v2/autofluxdep`` QubitFreqCfgTemplate.
    """


# --- default external bindings (project metadata can override) ---
def _default_qfw_factor() -> None:
    # First point has no previous linewidth/gain feedback; use the operator's
    # schema drive gain. Later points read the smoothed qfw_factor from the
    # orchestrator, mirroring notebook ``info.last.get("qfw_factor", ...)``.
    return None


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


def _resolve_drive_gain(
    mode: str,
    qfw_factor: Any | None,
    initial_gain: float,
    *,
    target_kappa: float,
    drive_gain_cap: float,
) -> float:
    if mode == _DRIVE_GAIN_MODE_ADAPTIVE:
        return _drive_gain_from_qfw_factor(
            qfw_factor,
            initial_gain,
            target_kappa=target_kappa,
            drive_gain_cap=drive_gain_cap,
        )
    if mode == _DRIVE_GAIN_MODE_FIXED:
        return float(initial_gain)
    raise RuntimeError(f"unsupported qubit_freq drive_gain_mode: {mode!r}")


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


def _qf_width_seed(ctx: Any | None) -> float | None:
    return ctx_md_float(ctx, "qf_w")


def _qfw_factor_seed(knobs: dict[str, Any]) -> float | None:
    width = knobs.get("qf_width_seed")
    if width is None:
        return None
    seed_gain = float(knobs["qub_gain"])
    if seed_gain <= 0.0:
        raise RuntimeError("qubit_freq qub_gain must be positive to seed qfw_factor")
    return float(width) / seed_gain


def _signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """PCA-rotate to the real axis and normalise to [0, 1] (a dip near 0)."""
    real = rotate2real(signals.astype(np.complex128)).real
    lo, hi = float(real.min()), float(real.max())
    real = (real - lo) / (hi - lo + 1e-12)
    # orient so the resonance is a dip (start/end high, centre low)
    if real[0] + real[-1] < real[len(real) // 2]:
        real = 1.0 - real
    return real


def _is_trusted_frequency_fit(
    real: NDArray[np.float64], fit_curve: NDArray[np.float64]
) -> bool:
    return is_good_fit(real, fit_curve, threshold=_FREQ_FIT_RESIDUAL_RATIO)


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

    return is_good_fit(real, fit_curve, threshold=_LINEWIDTH_FIT_RESIDUAL_RATIO)


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
        flux_device = require_flux_device(env, "qubit_freq")
        # cfg.dev is typed Mapping but make_cfg always populates it with a mutable
        # dict (GlobalDeviceManager.get_all_info); cast for the in-place name write.
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

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
        # liveplot settles round by round. The SNR probe + stop poll are threaded
        # into stop_checkers (early-stop on good SNR; cooperative cancel).
        probe = SnrProbe()
        stop_checkers = build_stop_checkers(env, probe, _signal2real)
        acquired = run_schedule_acquire(
            env=env,
            cfg=cfg,
            signal_shape=result.signal[idx].shape,
            dtype=np.complex128,
            configure_builder=lambda builder: builder.add(
                [
                    Pulse("init_pulse", cfg.modules.init_pulse, tag="init_pulse"),
                    Pulse("qubit_pulse", cfg.modules.qub_pulse, tag="qub_pulse"),
                    Readout("readout", cfg.modules.readout),
                ]
            ).declare_sweep("detune", detune_sweep),
            raw2signal_fn=acquire_to_complex,
            on_update=make_signal_update(
                result,
                idx,
                _signal2real,
                env.round_hook,
                probe=probe,
            ),
            program_cls=ModularProgramV2,
            stop_checkers=stop_checkers,
        )
        if acquired.stopped:
            return Patch()
        if acquired.signal is None:
            raise RuntimeError("qubit_freq Schedule acquire completed without signal")
        real = _signal2real(np.asarray(acquired.signal, dtype=np.complex128))

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
        knobs = env.knobs()
        recovery_reseeded = on_fit_succeeded(
            env,
            float(freq),
            snapshot_predict_freq=base_pred_qf,
            estimator_key=_PREDICT_FREQ_CORRECTION_SLOT.key,
        )
        base_for_residual = physical_prediction_for_residual(env, base_pred_qf, knobs)
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
        # orchestrator builds the SmoothingService from this declaration alone.
        Dependency("qfw_factor", smooth="step_weighted", default=_default_qfw_factor),
    )
    # the readout module: Node-produced → calibrated ml preset → default.
    optional_modules = (
        ModuleDep(
            "readout", default=missing_module_value, aliases=READOUT_LIBRARY_ALIASES
        ),
    )
    feedback_slots = (_PREDICT_FREQ_CORRECTION_SLOT,)

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Adapter-backed default cfg plus autofluxdep generation controls."""
        return adapter_node_schema(
            FreqAdapter,
            ctx,
            logical_paths={
                "qub_pulse": "modules.qub_pulse",
                "qub_ch": "modules.qub_pulse.ch",
                "qub_nqz": "modules.qub_pulse.nqz",
                "qub_gain": "modules.qub_pulse.gain",
                "qub_length": "modules.qub_pulse.waveform.length",
                "readout": "modules.readout",
                "relax_delay": "relax_delay",
                "reps": "reps",
                "rounds": "rounds",
                "detune_sweep": "sweep.freq",
            },
            generation_fields=(
                logical_generation_field(
                    "earlystop_snr",
                    FloatSpec(label="earlystop_snr", optional=True),
                    _DEFAULT_EARLYSTOP_SNR,
                    group="acquisition",
                ),
                acquire_retry_generation_field(),
                logical_generation_field(
                    "physical_recovery_mode",
                    str_choice_spec(
                        "mode",
                        (
                            PHYSICAL_RECOVERY_MODE_OFF,
                            PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT,
                        ),
                    ),
                    PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT,
                    group="freq_recovery",
                ),
                logical_generation_field(
                    "physical_recovery_min_points",
                    IntSpec(label="min_points"),
                    DEFAULT_PHYSICAL_RECOVERY_MIN_POINTS,
                    group="freq_recovery",
                ),
                logical_generation_field(
                    "physical_recovery_max_points",
                    IntSpec(label="max_points"),
                    DEFAULT_PHYSICAL_RECOVERY_MAX_POINTS,
                    group="freq_recovery",
                ),
                logical_generation_field(
                    "physical_recovery_max_center_shift_mhz",
                    FloatSpec(label="max_center_shift_mhz"),
                    DEFAULT_PHYSICAL_RECOVERY_MAX_CENTER_SHIFT_MHZ,
                    group="freq_recovery",
                ),
                logical_generation_field(
                    "physical_recovery_max_rms_mhz",
                    FloatSpec(label="max_rms_mhz"),
                    DEFAULT_PHYSICAL_RECOVERY_MAX_RMS_MHZ,
                    group="freq_recovery",
                ),
                logical_generation_field(
                    "drive_gain_mode",
                    str_choice_spec(
                        "mode",
                        (
                            _DRIVE_GAIN_MODE_ADAPTIVE,
                            _DRIVE_GAIN_MODE_FIXED,
                        ),
                    ),
                    _DRIVE_GAIN_MODE_ADAPTIVE,
                    group="drive_gain",
                ),
                logical_generation_field(
                    "target_kappa",
                    FloatSpec(label="target_kappa"),
                    _QFW_TARGET_KAPPA,
                    group="drive_gain",
                ),
                logical_generation_field(
                    "qf_width_seed",
                    FloatSpec(label="qf_width_seed", optional=True),
                    _qf_width_seed(ctx),
                    group="drive_gain",
                ),
                *feedback_generation_fields(
                    _PREDICT_FREQ_CORRECTION_SLOT,
                    group="predictor_correction",
                ),
            ),
            generation_choices=(
                generation_choice(
                    "freq_recovery",
                    "physical_recovery_mode",
                    {
                        PHYSICAL_RECOVERY_MODE_OFF: (),
                        PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT: (
                            "physical_recovery_min_points",
                            "physical_recovery_max_points",
                            "physical_recovery_max_center_shift_mhz",
                            "physical_recovery_max_rms_mhz",
                        ),
                    },
                ),
                generation_choice(
                    "drive_gain",
                    "drive_gain_mode",
                    {
                        _DRIVE_GAIN_MODE_FIXED: (),
                        _DRIVE_GAIN_MODE_ADAPTIVE: ("target_kappa", "qf_width_seed"),
                    },
                ),
                feedback_generation_choice(
                    _PREDICT_FREQ_CORRECTION_SLOT,
                    group="predictor_correction",
                ),
            ),
            default_overrides={"detune_sweep": _DEFAULT_DETUNE_SWEEP, "rounds": 10},
            drop_paths=("modules.reset",),
            module_ref_labels={"modules.readout": PULSE_READOUT_REF_LABELS},
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
        paths: list[OverridePath] = [
            OverridePath(
                "modules.qub_pulse.freq",
                "all_points",
                "predict_freq",
                "qubit drive frequency is generated from predictor feedback",
            )
        ]
        if knobs.get("drive_gain_mode") == _DRIVE_GAIN_MODE_ADAPTIVE:
            paths.append(
                OverridePath(
                    "modules.qub_pulse.gain",
                    "all_points",
                    "generation.drive_gain.drive_gain_mode",
                    "adaptive drive gain is generated from linewidth feedback",
                )
            )
        paths.extend(
            readout_module_override_paths(
                source="readout module dependency",
                reason="readout module is resolved from workflow/module-library dependency",
            )
        )
        return OverridePlan(tuple(paths))

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
        knobs = env.knobs()
        base_predict_freq = physical_prediction_for_make_cfg(
            env,
            float(snapshot["predict_freq"]),
            knobs,
        )
        predict_freq = base_predict_freq + _predict_freq_correction(env)
        qfw_factor = snapshot.get("qfw_factor")
        if qfw_factor is None:
            qfw_factor = _qfw_factor_seed(knobs)
        drive_gain = _resolve_drive_gain(
            str(knobs["drive_gain_mode"]),
            qfw_factor,
            float(knobs["qub_gain"]),
            target_kappa=float(knobs["target_kappa"]),
            drive_gain_cap=_DRIVE_GAIN_CAP,
        )
        patches: dict[str, object] = {
            "modules.qub_pulse.freq": predict_freq,
        }
        if str(knobs["drive_gain_mode"]) == _DRIVE_GAIN_MODE_ADAPTIVE:
            patches["modules.qub_pulse.gain"] = drive_gain
        patches.update(readout_module_patches(readout))
        raw_cfg = self.point_cfg(env, patches)
        raw_cfg.pop("sweep", None)
        return ml.make_cfg(raw_cfg, QubitFreqCfgTemplate)

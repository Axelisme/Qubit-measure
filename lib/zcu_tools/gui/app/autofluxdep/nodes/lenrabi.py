"""lenrabi — length-Rabi acquire and pi/pi2 pulse production.

The Builder lowers a qubit-frequency snapshot and readout module into the run
cfg. The short-lived Node applies flux, sweeps pulse length, fits the Rabi
trace, fills the Result row, and emits trusted scalar/module feedback.

- requires ``qubit_freq`` (a hard require via Dependency): the Rabi experiment
  drives the qubit on resonance, so no qubit frequency → no sensible cfg.
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).
- provides the ``pi_pulse`` module built from the fitted pi length, and provides
  ``pi2_pulse`` when the fitted pi2 length is finite and trusted.

``produce`` lowers the active context (a populated ml + an ``opt_readout`` module
+ the drive "設定頭" params) into a real ``LenRabiCfgTemplate`` via
``Builder.make_cfg`` → ``ml.make_cfg`` — mirroring the notebook's ``cfg_maker``
and the lower-layer ``experiment/v2/autofluxdep`` LenRabiCfgTemplate — then
acquires against a flux-aware MockSoc (offline) or real hardware. ``make_cfg``
Fast Fails when the context is unconfigured. Compare ``notebook_md/autofluxdep.md``
(the LenRabiTask block).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.gui.app.autofluxdep.cfg import (
    OverridePlan,
    SweepValue,
    pulse_module_ref_spec,
    pulse_readout_module_ref_spec,
)
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema, sweepcfg_to_axis
from zcu_tools.gui.app.autofluxdep.feedback import FeedbackSlotDecl
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    DEFAULT_ACQUIRE_RETRY,
    SnrProbe,
    acquire_retry,
    acquire_to_complex,
    axis_to_sweep,
    build_stop_condition,
    is_good_fit,
    make_signal_update,
    require_flux_device,
    set_flux_by_name,
    signal2real_flip,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.defaults import (
    pop_sweep_range,
    readout_module_patches,
)
from zcu_tools.gui.app.autofluxdep.nodes.dependency_defaults import (
    missing_info_value,
    missing_module_value,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import READOUT_LIBRARY_ALIASES
from zcu_tools.gui.app.autofluxdep.nodes.plotters import ColormapLinePlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep, Need
from zcu_tools.gui.app.autofluxdep.nodes.timing_defaults import (
    auto_relax_delay_from_t1,
    auto_stop_sweep_range,
    auto_sweep_stop,
    fixed_sweep_range,
    seed_md_float,
)
from zcu_tools.gui.app.autofluxdep.nodes.utils import (
    NodeOverridePlan,
    NodeSchemaBuilder,
    ctx_md_float,
    ctx_module,
    module_ref_default,
    module_ref_value_from_ctx,
    pulse_length,
    pulse_product,
)
from zcu_tools.gui.session.types import ExpContext
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_rabi

logger = logging.getLogger(__name__)

_DRIVE_GAIN_CAP = 1.0
_MIN_TRUSTED_PI_LENGTH = 0.03
_MAX_PI_SWEEP_FRACTION = 0.9
_MAX_FIT_RESIDUAL_RATIO = 0.1
_SWEEP_RANGE_MODE_AUTO_PI_LENGTH = "auto_pi_length"
_SWEEP_RANGE_MODE_FIXED = "fixed"
_RELAX_DELAY_MODE_AUTO_T1 = "auto_t1"
_RELAX_DELAY_MODE_FIXED = "fixed"
_DRIVE_GAIN_MODE_AUTO_PI_PRODUCT = "auto_pi_product"
_DRIVE_GAIN_MODE_FIXED = "fixed"
_PI_PULSE_SEED_NAMES = ("pi_len", "pi_amp", "pi_pulse")
_DRIVE_GAIN_SLOT = FeedbackSlotDecl(
    key="drive_gain",
    kind="controller",
    prefix="pi_gain_feedback",
    default_step_gain=0.5,
    default_decay_points=3.0,
)


class LenRabiModuleCfg(ConfigBase):
    """The module bundle lenrabi lowers a context into (mirrors the lower-layer
    ``experiment/v2/autofluxdep`` LenRabiModuleCfg without the unused reset): the
    on-resonance ``rabi_pulse`` (the swept drive) and the ``readout``."""

    rabi_pulse: PulseCfg
    readout: ReadoutCfg


class LenRabiCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base length-Rabi cfg lenrabi lowers a context into.

    ``ProgramV2Cfg`` (reps/rounds/relax) + the ``ExpCfgModel`` device/save fields
    + the ``rabi_pulse``/``readout`` modules + the ``sweep_range`` (the pulse-length
    extent as a ``(start, stop)`` pair) — mirroring the lower-layer
    ``experiment/v2/autofluxdep`` LenRabiCfgTemplate. The flux ``dev`` entry and the
    concrete ``length`` sweep are merged in by ``produce`` (the dev with this
    point's flux value, the length sweep over the Result axis); they are NOT part
    of the template, exactly like qubit_freq's detune sweep.
    """

    modules: LenRabiModuleCfg
    sweep_range: tuple[float, float]


def _last_fit(result: Any) -> float:
    """Return the last non-nan fit_value (the most recent pi_length)."""
    valid = result.fit_value[~np.isnan(result.fit_value)]
    return float(valid[-1]) if valid.size else float("nan")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _require_positive_finite(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise RuntimeError(f"lenrabi {name} must be positive and finite")
    return out


def _optional_positive_finite(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _require_positive_finite(name, value)


def _drive_gain_from_pi_product(
    pi_product: float, target_pi_length: float, *, factor: float, drive_gain_cap: float
) -> float:
    product = _require_positive_finite("pi_product", pi_product)
    target = _require_positive_finite("target_pi_length", target_pi_length)
    gain_factor = _require_positive_finite("pi_product_factor", factor)
    gain_cap = _require_positive_finite("drive_gain_cap", drive_gain_cap)
    return min(gain_cap, product / (gain_factor * target))


def _clamp_drive_gain(value: float) -> float:
    gain = _require_positive_finite("drive_gain", value)
    gain_cap = _DRIVE_GAIN_CAP
    return min(gain_cap, gain)


def _blend_positive(prior: float, feedback: float, confidence: float) -> float:
    prior_value = _require_positive_finite("prior drive_gain", prior)
    feedback_value = _require_positive_finite("feedback drive_gain", feedback)
    weight = float(confidence)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise RuntimeError("lenrabi feedback confidence must be between 0 and 1")
    return prior_value * math.exp(weight * math.log(feedback_value / prior_value))


def _drive_pulse_with_length(base: PulseCfg, length: float) -> dict[str, Any]:
    pulse = base.model_copy(deep=True)
    pulse.set_param("length", float(length))
    return pulse.to_dict()


@dataclass(frozen=True)
class _LenRabiFeedbackInputs:
    target_pi_length: float
    range_pi_length: float
    feedback_pi_product: float


def _resolve_feedback_inputs(
    snapshot: Snapshot, knobs: dict[str, Any]
) -> _LenRabiFeedbackInputs:
    target_pi_length = _require_positive_finite(
        "expected_pi_length", knobs["expected_pi_length"]
    )
    range_pi_length = (
        _optional_positive_finite("previous pi_length", snapshot.get("pi_length"))
        or target_pi_length
    )
    feedback_pi_product = _optional_positive_finite(
        "previous pi_product", snapshot.get("pi_product")
    ) or _require_positive_finite("pi_product_seed", knobs["pi_product_seed"])
    return _LenRabiFeedbackInputs(
        target_pi_length=target_pi_length,
        range_pi_length=range_pi_length,
        feedback_pi_product=feedback_pi_product,
    )


@dataclass(frozen=True)
class _LenRabiFit:
    pi_length: float
    pi2_length: float
    rabi_freq: float
    residual: float
    fit_curve: NDArray[np.float64]


def _fit_residual_sort_key(fit: _LenRabiFit) -> tuple[int, float]:
    if not np.isfinite(fit.residual):
        return (1, np.inf)
    return (0, fit.residual)


def _fit_lenrabi(
    lengths: NDArray[np.float64], real: NDArray[np.float64]
) -> _LenRabiFit:
    fits: list[_LenRabiFit] = []
    fit_errors: list[RuntimeError] = []
    for decay in (True, False):
        try:
            pi_x, _, pi2_x, _, freq, _, fit_curve, _ = fit_rabi(
                lengths, real, decay=decay
            )
        except RuntimeError as exc:
            fit_errors.append(exc)
            logger.debug(
                "lenrabi %s fit candidate failed: %s",
                "decay" if decay else "non-decay",
                exc,
            )
            continue
        curve = np.asarray(fit_curve, dtype=np.float64)
        residual = float(np.mean(np.abs(np.asarray(real, dtype=np.float64) - curve)))
        fits.append(
            _LenRabiFit(
                pi_length=float(pi_x),
                pi2_length=float(pi2_x),
                rabi_freq=float(freq),
                residual=residual,
                fit_curve=curve,
            )
        )
    if not fits:
        raise RuntimeError(
            "lenrabi fit failed for decay and non-decay candidates"
        ) from (fit_errors[-1] if fit_errors else None)
    return min(fits, key=_fit_residual_sort_key)


def _is_trusted_lenrabi_fit(
    fit: _LenRabiFit, lengths: NDArray[np.float64], real: NDArray[np.float64]
) -> bool:
    if not (
        np.isfinite(fit.pi_length)
        and np.isfinite(fit.rabi_freq)
        and np.isfinite(fit.residual)
    ):
        return False
    if fit.pi_length < _MIN_TRUSTED_PI_LENGTH:
        return False
    max_length = float(np.max(np.asarray(lengths, dtype=np.float64)))
    if (
        not np.isfinite(max_length)
        or fit.pi_length > _MAX_PI_SWEEP_FRACTION * max_length
    ):
        return False
    return is_good_fit(real, fit.fit_curve, threshold=_MAX_FIT_RESIDUAL_RATIO)


def _fill_lenrabi_fit_or_skip(
    result: Any,
    idx: int,
    lengths: NDArray[np.float64],
    real: NDArray[np.float64],
    fit: _LenRabiFit,
    round_hook: Callable[[int], None] | None,
) -> bool:
    if not _is_trusted_lenrabi_fit(fit, lengths, real):
        logger.debug("lenrabi fit @flux%d: untrusted fit discarded", idx)
        if round_hook is not None:
            round_hook(idx)
        return False
    result.fit_value[idx] = fit.pi_length
    np.copyto(result.fit_curve[idx], fit.fit_curve)
    if round_hook is not None:
        round_hook(idx)
    return True


def _patch_from_lenrabi_fit(fit: _LenRabiFit, base_drive_pulse: PulseCfg) -> Patch:
    patch = Patch()
    patch.set("pi_length", fit.pi_length)
    patch.set("rabi_freq", fit.rabi_freq)
    patch.set("pi_product", fit.pi_length * float(base_drive_pulse.gain))
    pi2_length = float(fit.pi2_length)
    if not (np.isfinite(pi2_length) and pi2_length >= _MIN_TRUSTED_PI_LENGTH):
        return patch
    patch.set("pi2_length", pi2_length)
    patch.set_module(
        "pi_pulse", _drive_pulse_with_length(base_drive_pulse, fit.pi_length)
    )
    patch.set_module(
        "pi2_pulse", _drive_pulse_with_length(base_drive_pulse, pi2_length)
    )
    return patch


def _update_drive_gain_feedback(
    env: RunEnv, fit: _LenRabiFit, base_drive_pulse: PulseCfg
) -> None:
    if env.feedback is None:
        return
    knobs = env.knobs()
    if str(knobs["drive_gain_mode"]) != _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT:
        return
    controller = env.feedback.controller(_DRIVE_GAIN_SLOT.key)
    if controller is None:
        return
    target = _require_positive_finite("expected_pi_length", knobs["expected_pi_length"])
    current_gain = _require_positive_finite("drive_gain", base_drive_pulse.gain)
    normalized_error = math.log(
        _require_positive_finite("pi_length", fit.pi_length) / target
    )
    controller.propose(current_gain, normalized_error)


class LenRabiNode(Node):
    """One flux point's lenrabi: set flux → real acquire → fit_rabi → fill row → Patch.

    Mirrors the lower-layer LenRabi Schedule acquire + ``run``: the on-resonance
    drive sweeps its pulse length, ``ModularProgramV2`` (rabi_pulse → Readout)
    acquires per round, and ``fit_rabi`` recovers the pi / pi2 lengths + Rabi freq.
    """

    def __init__(self, env: RunEnv, builder: LenRabiBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

        result: Sweep1DResult = env.result
        idx = env.flux_idx

        # Lower the active context into the run cfg (Fast Fail if unconfigured: a
        # real acquire needs a concrete readout + on-resonance drive pulse).
        cfg = self._builder.make_cfg(env, snapshot)
        lo, hi = cfg.sweep_range
        lengths = np.linspace(float(lo), float(hi), result.n_x)
        result.x[:] = lengths

        # Point the flux device at this sweep point and push it to hardware (mock:
        # writes the FakeDevice value → SimEngine reads it live).
        flux_device = require_flux_device(env, "lenrabi")
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

        # Sweep the rabi pulse length over the Result's trailing axis (the lower
        # layer's sweep2param("length") set on rabi_pulse).
        length_sweep = axis_to_sweep(lengths)
        length_param = sweep2param("length", length_sweep)
        base_drive_pulse = cfg.modules.rabi_pulse.model_copy(deep=True)
        cfg.modules.rabi_pulse.set_param("length", length_param)

        result.flux[idx] = env.flux

        probe = SnrProbe()
        signal_buffer = SignalBuffer(
            result.signal[idx].shape,
            dtype=np.complex128,
            on_update=make_signal_update(
                result,
                idx,
                signal2real_flip,
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
                    Pulse("rabi_pulse", cfg.modules.rabi_pulse),
                    Readout("readout", cfg.modules.readout),
                ]
            ).declare_sweep("length", length_sweep)
            signal = builder.build_and_acquire(
                raw2signal_fn=acquire_to_complex,
                retry=acquire_retry(env),
                progress=False,
                progress_label=f"{env.node_name or 'lenrabi'} flux {idx + 1} rounds",
                progress_leave=False,
                stop_condition=build_stop_condition(env, probe, signal2real_flip),
            )
            outcome = sched.outcome

        if outcome.status == "stopped":
            return Patch()
        if outcome.status == "failed":
            reason = outcome.reason or "lenrabi Schedule acquire failed"
            raise RuntimeError(reason) from outcome.exception
        if outcome.status == "interrupted":
            reason = outcome.reason or "lenrabi Schedule acquire interrupted"
            raise RuntimeError(reason) from outcome.exception
        if outcome.status != "completed":
            raise RuntimeError(
                f"unsupported lenrabi Schedule outcome: {outcome.status!r}"
            )

        real = signal2real_flip(np.asarray(signal, dtype=np.complex128))

        fit = _fit_lenrabi(lengths, real)

        # The fitted single scalar (the Result's fit_value) is the pi length; the
        # extra Patch keys/modules below are lenrabi-specific and stay in the node.
        if not _fill_lenrabi_fit_or_skip(
            result, idx, lengths, real, fit, env.round_hook
        ):
            # partial: omit feedback values/modules so downstream keeps fallback
            return Patch()

        logger.debug(
            "lenrabi fit @flux%d: rabi_freq=%.4f pi_len=%.3f pi2_len=%.3f",
            idx,
            fit.rabi_freq,
            fit.pi_length,
            fit.pi2_length,
        )
        _update_drive_gain_feedback(env, fit, base_drive_pulse)

        return _patch_from_lenrabi_fit(fit, base_drive_pulse)


class LenRabiBuilder(Builder):
    """The lenrabi provider — acquire Rabi oscillation, real fit_rabi, accumulating
    colormap. Produces scalar pi_length / pi2_length / rabi_freq info values, and
    only produces the pi_pulse / pi2_pulse pair when the fitted pi2 length is trusted.
    """

    name = "lenrabi"
    provides = ("pi_length", "pi2_length", "rabi_freq", "pi_product")
    provides_modules = ("pi_pulse", "pi2_pulse")
    requires = (Dependency("qubit_freq", need=Need.NOW),)
    optional = (
        Dependency("t1", smooth="ewma", default=missing_info_value),
        Dependency("pi_length", default=missing_info_value),
        Dependency("pi_product", smooth="step_weighted", default=missing_info_value),
    )
    optional_modules = (
        ModuleDep(
            "opt_readout", default=missing_module_value, aliases=READOUT_LIBRARY_ALIASES
        ),
    )
    feedback_slots = (_DRIVE_GAIN_SLOT,)

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Default cfg plus autofluxdep generation controls."""
        t1_seed = seed_md_float(ctx, "t1", 10.0)
        pi_seed_module = ctx_module(ctx, *_PI_PULSE_SEED_NAMES)
        pi_len_seed = ctx_md_float(ctx, "pi_len") or pulse_length(pi_seed_module) or 1.0
        pi_product_seed = pulse_product(pi_seed_module) or pi_len_seed
        relax_factor = 3.0
        relax_min_us = 0.0
        sweep_start_us = 0.05
        sweep_stop_factor = 5.0

        qub_ch = 0
        if isinstance(ctx, ExpContext):
            value = ctx.md.get("qub_ch")
            if isinstance(value, int) and not isinstance(value, bool):
                qub_ch = value

        rabi_pulse_spec = pulse_module_ref_spec(label="Rabi Pulse")
        readout_spec = pulse_readout_module_ref_spec(label="Readout")
        rabi_pulse_default = module_ref_value_from_ctx(
            ctx, *_PI_PULSE_SEED_NAMES, accepted_types=("pulse",)
        )
        if rabi_pulse_default is None:
            rabi_pulse_default = module_ref_value_from_ctx(
                ctx, "rabi_pulse", accepted_types=("pulse",)
            )
        if rabi_pulse_default is None:
            rabi_pulse_default = module_ref_default(ctx, rabi_pulse_spec)
            if rabi_pulse_default is None:
                raise RuntimeError("lenrabi pulse default is unexpectedly empty")
            rabi_pulse_default.with_field("ch", qub_ch)
            rabi_pulse_default.with_field("gain", 0.3)
        rabi_pulse_default.with_field("waveform.length", 1.0)

        return (
            NodeSchemaBuilder(label="Length Rabi")
            .field(
                "rabi_pulse",
                "modules.rabi_pulse",
                spec=rabi_pulse_spec,
                default=rabi_pulse_default,
            )
            .field(
                "readout",
                "modules.readout",
                spec=readout_spec,
                default=module_ref_default(
                    ctx,
                    readout_spec,
                    *READOUT_LIBRARY_ALIASES,
                    accepted_types=("readout/pulse",),
                ),
            )
            .float(
                "relax_delay",
                "relax_delay",
                label="Relax delay (us)",
                default=auto_relax_delay_from_t1(
                    t1_seed,
                    factor=relax_factor,
                    minimum=relax_min_us,
                ),
                decimals=3,
            )
            .sweep(
                "sweep_range",
                "sweep.length",
                label="Length (us)",
                default=SweepValue(
                    *auto_stop_sweep_range(
                        pi_len_seed,
                        start=sweep_start_us,
                        stop_factor=sweep_stop_factor,
                        stop_min=None,
                    ),
                    expts=101,
                ),
            )
            .int("reps", "reps", label="Reps", default=1000)
            .int("rounds", "rounds", label="Rounds", default=10)
            .logical("qub_ch", "modules.rabi_pulse.ch")
            .logical("qub_nqz", "modules.rabi_pulse.nqz")
            .logical("qub_gain", "modules.rabi_pulse.gain")
            .acquisition(earlystop_snr=30.0, acquire_retry=DEFAULT_ACQUIRE_RETRY)
            .auto_relax_from_t1(
                seed_us=t1_seed,
                factor=relax_factor,
                minimum_us=relax_min_us,
            )
            .choice(
                "sweep_range_mode",
                "generation.sweep.sweep_range_mode",
                label="range_mode",
                choices=(_SWEEP_RANGE_MODE_AUTO_PI_LENGTH, _SWEEP_RANGE_MODE_FIXED),
                default=_SWEEP_RANGE_MODE_AUTO_PI_LENGTH,
                tooltip=(
                    "Auto derives the Rabi sweep stop from pi-length feedback; "
                    "start/expts stay in Default cfg."
                ),
            )
            .float(
                "expected_pi_length",
                "generation.sweep.expected_pi_length",
                label="target_pi_length_us",
                default=pi_len_seed,
                tooltip="Pi-length setpoint for drive gain feedback.",
            )
            .float(
                "sweep_stop_factor",
                "generation.sweep.sweep_stop_factor",
                label="stop_factor",
                default=sweep_stop_factor,
                tooltip="Pi-length multiplier for the auto sweep stop.",
            )
            .choice_group(
                "generation.sweep",
                "sweep_range_mode",
                {
                    _SWEEP_RANGE_MODE_FIXED: (),
                    _SWEEP_RANGE_MODE_AUTO_PI_LENGTH: ("sweep_stop_factor",),
                },
            )
            .choice(
                "drive_gain_mode",
                "generation.drive_gain.drive_gain_mode",
                label="mode",
                choices=(_DRIVE_GAIN_MODE_AUTO_PI_PRODUCT, _DRIVE_GAIN_MODE_FIXED),
                default=_DRIVE_GAIN_MODE_AUTO_PI_PRODUCT,
                tooltip="Auto uses pi-product history; fixed keeps Default cfg gain.",
            )
            .float(
                "pi_product_seed",
                "generation.drive_gain.pi_product_seed",
                label="initial_pi_product",
                default=pi_product_seed,
                tooltip="Initial pi-length times gain before feedback exists.",
            )
            .choice_group(
                "generation.drive_gain",
                "drive_gain_mode",
                {
                    _DRIVE_GAIN_MODE_FIXED: (),
                    _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT: ("pi_product_seed",),
                },
            )
            .feedback_slot(_DRIVE_GAIN_SLOT, group="pi_feedback")
            .build()
        )

    def make_init_result(
        self, schema: NodeCfgSchema, flux: Any, md: Any = None
    ) -> Sweep1DResult:
        knobs = schema.lower(None, md=md)
        lengths = sweepcfg_to_axis(knobs["sweep_range"])
        return Sweep1DResult.allocate(flux, lengths, x_label="pulse length (us)")

    def make_plotter(self, figure: Any) -> ColormapLinePlotter:
        return ColormapLinePlotter(
            figure,
            title="lenrabi",
            y_label="Pulse length (us)",
            num_lines=3,
            marker_of=_last_fit,
        )

    def build_node(self, env: RunEnv) -> LenRabiNode:
        return LenRabiNode(env, self)

    def override_plan(self, schema: NodeCfgSchema) -> OverridePlan:
        knobs = schema.read_knobs()
        plan = NodeOverridePlan()
        plan.generated_if(
            True,
            "modules.rabi_pulse.freq",
            source="qubit_freq",
            reason="rabi pulse frequency is generated from qubit_freq dependency",
        )
        plan.generated_if(
            knobs.get("drive_gain_mode") == _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT,
            "modules.rabi_pulse.gain",
            source="generation.drive_gain.drive_gain_mode",
            reason="rabi drive gain is generated from pi-length feedback",
        )
        plan.generated_if(
            knobs.get("relax_delay_mode") == _RELAX_DELAY_MODE_AUTO_T1,
            "relax_delay",
            source="generation.relax.relax_delay_mode",
            reason="relax delay is generated from T1 feedback",
        )
        plan.generated_if(
            knobs.get("sweep_range_mode") == _SWEEP_RANGE_MODE_AUTO_PI_LENGTH,
            "sweep.length.stop",
            source="generation.sweep.sweep_range_mode",
            reason="Rabi sweep stop is generated from pi-length feedback",
        )
        plan.readout_dependency(source="opt_readout module dependency")
        return plan.build()

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> LenRabiCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the notebook's lenrabi ``cfg_maker`` (runs in ``produce``, where
        the snapshot is available): the ``rabi_pulse`` drives the qubit on
        resonance — its frequency is the required ``qubit_freq`` from the snapshot —
        the readout is the latest-available ``opt_readout`` module, and the pulse
        waveform / channel / gain / nqz come from the node's params (the "設定頭").
        The pulse-length ``sweep_range`` is generated from the latest or seeded pi
        length by default (notebook: ``(0.05, max(5*prev_pi_len, 0.5))``). The flux
        ``dev`` entry and the concrete ``length`` sweep are NOT here — ``produce``
        merges them, exactly like qubit_freq's detune.

        Raises if the readout module is unavailable or the drive params are unset —
        a real run needs a concrete drive pulse (Fast Fail).
        """
        ml = env.ml
        if ml is None:
            raise RuntimeError("lenrabi.make_cfg needs an active ModuleLibrary")
        readout = snapshot.module("opt_readout")
        if readout is None:
            raise RuntimeError(
                "lenrabi.make_cfg needs a readout module (none produced or preset)"
            )
        knobs = env.knobs()
        qubit_freq = float(snapshot["qubit_freq"])
        t1 = _float_or_none(snapshot.get("t1")) or float(knobs["t1_seed_us"])
        feedback = _resolve_feedback_inputs(snapshot, knobs)

        drive_gain_mode = str(knobs["drive_gain_mode"])
        if drive_gain_mode == _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT:
            drive_gain = _drive_gain_from_pi_product(
                feedback.feedback_pi_product,
                feedback.target_pi_length,
                factor=1.2,
                drive_gain_cap=_DRIVE_GAIN_CAP,
            )
        elif drive_gain_mode == _DRIVE_GAIN_MODE_FIXED:
            drive_gain = float(knobs["qub_gain"])
        else:
            raise RuntimeError(
                f"unsupported lenrabi drive_gain_mode: {drive_gain_mode!r}"
            )
        if (
            env.feedback is not None
            and drive_gain_mode == _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT
        ):
            controller = env.feedback.controller(_DRIVE_GAIN_SLOT.key)
            if controller is not None:
                latest = controller.latest()
                if latest is not None:
                    drive_gain = _blend_positive(
                        drive_gain,
                        latest.value,
                        latest.confidence,
                    )
                    drive_gain = _clamp_drive_gain(drive_gain)
        patches: dict[str, object] = {
            "modules.rabi_pulse.freq": qubit_freq,
        }
        if drive_gain_mode == _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT:
            patches["modules.rabi_pulse.gain"] = drive_gain
        patches.update(readout_module_patches(readout))

        relax_delay_mode = str(knobs["relax_delay_mode"])
        if relax_delay_mode == _RELAX_DELAY_MODE_AUTO_T1:
            relax_delay = auto_relax_delay_from_t1(
                t1,
                factor=float(knobs["relax_factor"]),
                minimum=float(knobs["relax_min_us"]),
            )
            patches["relax_delay"] = relax_delay
        elif relax_delay_mode == _RELAX_DELAY_MODE_FIXED:
            relax_delay = float(knobs["relax_delay"])
        else:
            raise RuntimeError(
                f"unsupported lenrabi relax_delay_mode: {relax_delay_mode!r}"
            )

        sweep_range_mode = str(knobs["sweep_range_mode"])
        if sweep_range_mode == _SWEEP_RANGE_MODE_AUTO_PI_LENGTH:
            fixed_sweep = knobs["sweep_range"]
            sweep_range = (
                float(fixed_sweep.start),
                auto_sweep_stop(
                    feedback.range_pi_length,
                    stop_factor=float(knobs["sweep_stop_factor"]),
                    stop_min=None,
                ),
            )
            patches["sweep.length.stop"] = sweep_range[1]
        elif sweep_range_mode == _SWEEP_RANGE_MODE_FIXED:
            sweep_range = fixed_sweep_range(knobs["sweep_range"])
        else:
            raise RuntimeError(
                f"unsupported lenrabi sweep_range_mode: {sweep_range_mode!r}"
            )
        raw_cfg = self.point_cfg(env, patches)
        raw_cfg["sweep_range"] = pop_sweep_range(raw_cfg, "length", node_name=self.name)
        return ml.make_cfg(raw_cfg, LenRabiCfgTemplate)

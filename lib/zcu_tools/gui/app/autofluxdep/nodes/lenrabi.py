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
from zcu_tools.experiment.v2_gui.adapters.twotone.rabi.len_rabi import LenRabiAdapter
from zcu_tools.gui.app.autofluxdep.cfg import (
    FloatSpec,
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
    build_stop_condition,
    is_good_fit,
    make_signal_update,
    require_flux_device,
    run_schedule_acquire,
    set_flux_by_name,
    signal2real_flip,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.defaults import (
    PULSE_READOUT_REF_LABELS,
    adapter_node_schema,
    ctx_md_float,
    ctx_module,
    generation_choice,
    logical_generation_field,
    pop_sweep_range,
    pulse_length,
    pulse_product,
    readout_module_override_paths,
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
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.timing_defaults import (
    auto_relax_delay_from_t1,
    auto_stop_sweep_range,
    fixed_sweep_range,
    seed_md_float,
)
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

_DEFAULT_EARLYSTOP_SNR = 30.0
_DEFAULT_T1 = 10.0
_DEFAULT_EXPECTED_PI_LENGTH = 1.0
_DEFAULT_SWEEP_START = 0.05
_DEFAULT_SWEEP_STOP_FACTOR = 5.0
_DEFAULT_SWEEP_STOP_MIN = 0.5
_DEFAULT_RELAX_FACTOR = 3.0
_DEFAULT_RELAX_MIN = 0.0
_DEFAULT_PI_PRODUCT_FACTOR = 1.2
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


def _seed_t1(ctx: Any | None) -> float:
    return seed_md_float(ctx, "t1", _DEFAULT_T1)


def _seed_pi_length(ctx: Any | None) -> float:
    md_value = ctx_md_float(ctx, "pi_len")
    if md_value is not None:
        return md_value
    module = ctx_module(ctx, "pi_amp", "pi_len", "pi_pulse")
    return pulse_length(module) or _DEFAULT_EXPECTED_PI_LENGTH


def _seed_pi_product(ctx: Any | None) -> float:
    module = ctx_module(ctx, "pi_amp", "pi_len", "pi_pulse")
    return pulse_product(module) or _seed_pi_length(ctx)


def _resolve_cfg_sweep_range(
    mode: str, *, pi_length: float, fixed: Any, knobs: dict[str, Any]
) -> tuple[float, float]:
    if mode == _SWEEP_RANGE_MODE_AUTO_PI_LENGTH:
        return auto_stop_sweep_range(
            pi_length,
            start=float(knobs["sweep_start_us"]),
            stop_factor=float(knobs["sweep_stop_factor"]),
            stop_min=float(knobs["sweep_stop_min_us"]),
        )
    if mode == _SWEEP_RANGE_MODE_FIXED:
        return fixed_sweep_range(fixed)
    raise RuntimeError(f"unsupported lenrabi sweep_range_mode: {mode!r}")


def _resolve_cfg_relax_delay(
    mode: str, *, t1: float, fixed: float, knobs: dict[str, Any]
) -> float:
    if mode == _RELAX_DELAY_MODE_AUTO_T1:
        return auto_relax_delay_from_t1(
            t1,
            factor=float(knobs["relax_factor"]),
            minimum=float(knobs["relax_min_us"]),
        )
    if mode == _RELAX_DELAY_MODE_FIXED:
        return float(fixed)
    raise RuntimeError(f"unsupported lenrabi relax_delay_mode: {mode!r}")


def _drive_gain_from_pi_product(
    pi_product: float, target_pi_length: float, *, factor: float, drive_gain_cap: float
) -> float:
    product = _require_positive_finite("pi_product", pi_product)
    target = _require_positive_finite("target_pi_length", target_pi_length)
    gain_factor = _require_positive_finite("pi_product_factor", factor)
    gain_cap = _require_positive_finite("drive_gain_cap", drive_gain_cap)
    return min(gain_cap, product / (gain_factor * target))


def _resolve_drive_gain(
    mode: str,
    *,
    pi_product: float,
    target_pi_length: float,
    fixed: float,
) -> float:
    if mode == _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT:
        return _drive_gain_from_pi_product(
            pi_product,
            target_pi_length,
            factor=_DEFAULT_PI_PRODUCT_FACTOR,
            drive_gain_cap=_DRIVE_GAIN_CAP,
        )
    if mode == _DRIVE_GAIN_MODE_FIXED:
        return float(fixed)
    raise RuntimeError(f"unsupported lenrabi drive_gain_mode: {mode!r}")


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
        stop_condition = build_stop_condition(env, probe, signal2real_flip)
        acquired = run_schedule_acquire(
            env=env,
            cfg=cfg,
            signal_shape=result.signal[idx].shape,
            dtype=np.complex128,
            configure_builder=lambda builder: builder.add(
                [
                    Pulse("rabi_pulse", cfg.modules.rabi_pulse),
                    Readout("readout", cfg.modules.readout),
                ]
            ).declare_sweep("length", length_sweep),
            raw2signal_fn=acquire_to_complex,
            on_update=make_signal_update(
                result,
                idx,
                signal2real_flip,
                env.round_hook,
                probe=probe,
            ),
            program_cls=ModularProgramV2,
            stop_condition=stop_condition,
        )
        if acquired.stopped:
            return Patch()
        if acquired.signal is None:
            raise RuntimeError("lenrabi Schedule acquire completed without signal")
        real = signal2real_flip(np.asarray(acquired.signal, dtype=np.complex128))

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
    requires = (Dependency("qubit_freq"),)
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
        """Adapter-backed default cfg plus autofluxdep generation controls."""
        t1_seed = _seed_t1(ctx)
        pi_len_seed = _seed_pi_length(ctx)
        return adapter_node_schema(
            LenRabiAdapter,
            ctx,
            logical_paths={
                "rabi_pulse": "modules.rabi_pulse",
                "qub_ch": "modules.rabi_pulse.ch",
                "qub_nqz": "modules.rabi_pulse.nqz",
                "qub_gain": "modules.rabi_pulse.gain",
                "readout": "modules.readout",
                "relax_delay": "relax_delay",
                "reps": "reps",
                "rounds": "rounds",
                "sweep_range": "sweep.length",
            },
            generation_fields=(
                logical_generation_field(
                    "earlystop_snr",
                    FloatSpec(
                        label="earlystop_snr",
                        optional=True,
                        tooltip="Stop averaging once completed-round SNR reaches this value.",
                    ),
                    _DEFAULT_EARLYSTOP_SNR,
                    group="acquisition",
                ),
                acquire_retry_generation_field(),
                logical_generation_field(
                    "relax_delay_mode",
                    str_choice_spec(
                        "delay_mode",
                        (_RELAX_DELAY_MODE_AUTO_T1, _RELAX_DELAY_MODE_FIXED),
                        tooltip="Auto derives relax delay from T1; fixed keeps Default cfg delay.",
                    ),
                    _RELAX_DELAY_MODE_AUTO_T1,
                    group="relax",
                ),
                logical_generation_field(
                    "t1_seed_us",
                    FloatSpec(
                        label="initial_t1_us",
                        tooltip="Initial T1 before measured feedback exists.",
                    ),
                    t1_seed,
                    group="relax",
                ),
                logical_generation_field(
                    "relax_factor",
                    FloatSpec(
                        label="factor",
                        tooltip="Multiplier applied to T1 for auto relax delay.",
                    ),
                    _DEFAULT_RELAX_FACTOR,
                    group="relax",
                ),
                logical_generation_field(
                    "relax_min_us",
                    FloatSpec(
                        label="min_us",
                        tooltip="Minimum auto relax delay.",
                    ),
                    _DEFAULT_RELAX_MIN,
                    group="relax",
                ),
                logical_generation_field(
                    "sweep_range_mode",
                    str_choice_spec(
                        "range_mode",
                        (_SWEEP_RANGE_MODE_AUTO_PI_LENGTH, _SWEEP_RANGE_MODE_FIXED),
                        tooltip="Auto derives the Rabi sweep from pi-length feedback.",
                    ),
                    _SWEEP_RANGE_MODE_AUTO_PI_LENGTH,
                    group="sweep",
                    group_label="Rabi sweep window",
                ),
                logical_generation_field(
                    "expected_pi_length",
                    FloatSpec(
                        label="target_pi_length_us",
                        tooltip="Pi-length setpoint for drive gain feedback.",
                    ),
                    pi_len_seed,
                    group="sweep",
                ),
                logical_generation_field(
                    "sweep_start_us",
                    FloatSpec(
                        label="start_us",
                        tooltip="Lower bound for the auto Rabi sweep.",
                    ),
                    _DEFAULT_SWEEP_START,
                    group="sweep",
                ),
                logical_generation_field(
                    "sweep_stop_factor",
                    FloatSpec(
                        label="stop_factor",
                        tooltip="Pi-length multiplier for the auto sweep stop.",
                    ),
                    _DEFAULT_SWEEP_STOP_FACTOR,
                    group="sweep",
                ),
                logical_generation_field(
                    "sweep_stop_min_us",
                    FloatSpec(
                        label="stop_min_us",
                        tooltip="Minimum stop value for the auto Rabi sweep.",
                    ),
                    _DEFAULT_SWEEP_STOP_MIN,
                    group="sweep",
                ),
                logical_generation_field(
                    "drive_gain_mode",
                    str_choice_spec(
                        "mode",
                        (_DRIVE_GAIN_MODE_AUTO_PI_PRODUCT, _DRIVE_GAIN_MODE_FIXED),
                        tooltip="Auto uses pi-product history; fixed keeps Default cfg gain.",
                    ),
                    _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT,
                    group="drive_gain",
                    group_label="Drive-gain baseline",
                ),
                logical_generation_field(
                    "pi_product_seed",
                    FloatSpec(
                        label="initial_pi_product",
                        tooltip="Initial pi-length times gain before feedback exists.",
                    ),
                    _seed_pi_product(ctx),
                    group="drive_gain",
                ),
                *feedback_generation_fields(_DRIVE_GAIN_SLOT, group="pi_feedback"),
            ),
            generation_choices=(
                generation_choice(
                    "relax",
                    "relax_delay_mode",
                    {
                        _RELAX_DELAY_MODE_FIXED: (),
                        _RELAX_DELAY_MODE_AUTO_T1: (
                            "t1_seed_us",
                            "relax_factor",
                            "relax_min_us",
                        ),
                    },
                ),
                generation_choice(
                    "sweep",
                    "sweep_range_mode",
                    {
                        _SWEEP_RANGE_MODE_FIXED: (),
                        _SWEEP_RANGE_MODE_AUTO_PI_LENGTH: (
                            "sweep_start_us",
                            "sweep_stop_factor",
                            "sweep_stop_min_us",
                        ),
                    },
                ),
                generation_choice(
                    "drive_gain",
                    "drive_gain_mode",
                    {
                        _DRIVE_GAIN_MODE_FIXED: (),
                        _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT: ("pi_product_seed",),
                    },
                ),
                feedback_generation_choice(_DRIVE_GAIN_SLOT, group="pi_feedback"),
            ),
            default_overrides={
                "rounds": 10,
                "relax_delay": auto_relax_delay_from_t1(
                    t1_seed,
                    factor=_DEFAULT_RELAX_FACTOR,
                    minimum=_DEFAULT_RELAX_MIN,
                ),
                "sweep_range": SweepValue(
                    *auto_stop_sweep_range(
                        pi_len_seed,
                        start=_DEFAULT_SWEEP_START,
                        stop_factor=_DEFAULT_SWEEP_STOP_FACTOR,
                        stop_min=_DEFAULT_SWEEP_STOP_MIN,
                    ),
                    expts=101,
                ),
            },
            path_renames={"modules.qub_pulse": "modules.rabi_pulse"},
            drop_paths=("modules.reset",),
            module_ref_labels={"modules.readout": PULSE_READOUT_REF_LABELS},
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
        paths: list[OverridePath] = [
            OverridePath(
                "modules.rabi_pulse.freq",
                "all_points",
                "qubit_freq",
                "rabi pulse frequency is generated from qubit_freq dependency",
            )
        ]
        if knobs.get("drive_gain_mode") == _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT:
            paths.append(
                OverridePath(
                    "modules.rabi_pulse.gain",
                    "all_points",
                    "generation.drive_gain.drive_gain_mode",
                    "rabi drive gain is generated from pi-length feedback",
                )
            )
        if knobs.get("relax_delay_mode") == _RELAX_DELAY_MODE_AUTO_T1:
            paths.append(
                OverridePath(
                    "relax_delay",
                    "all_points",
                    "generation.relax.relax_delay_mode",
                    "relax delay is generated from T1 feedback",
                )
            )
        if knobs.get("sweep_range_mode") == _SWEEP_RANGE_MODE_AUTO_PI_LENGTH:
            paths.append(
                OverridePath(
                    "sweep.length",
                    "all_points",
                    "generation.sweep.sweep_range_mode",
                    "Rabi sweep range is generated from pi-length feedback",
                )
            )
        paths.extend(
            readout_module_override_paths(
                source="opt_readout module dependency",
                reason="readout module is resolved from workflow/module-library dependency",
            )
        )
        return OverridePlan(tuple(paths))

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
        drive_gain = _resolve_drive_gain(
            drive_gain_mode,
            pi_product=feedback.feedback_pi_product,
            target_pi_length=feedback.target_pi_length,
            fixed=float(knobs["qub_gain"]),
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
        relax_delay = _resolve_cfg_relax_delay(
            str(knobs["relax_delay_mode"]),
            t1=t1,
            fixed=float(knobs["relax_delay"]),
            knobs=knobs,
        )
        if str(knobs["relax_delay_mode"]) == _RELAX_DELAY_MODE_AUTO_T1:
            patches["relax_delay"] = relax_delay
        sweep_range = _resolve_cfg_sweep_range(
            str(knobs["sweep_range_mode"]),
            pi_length=feedback.range_pi_length,
            fixed=knobs["sweep_range"],
            knobs=knobs,
        )
        if str(knobs["sweep_range_mode"]) == _SWEEP_RANGE_MODE_AUTO_PI_LENGTH:
            patches["sweep.length"] = sweep_range
        raw_cfg = self.point_cfg(env, patches)
        raw_cfg["sweep_range"] = pop_sweep_range(raw_cfg, "length", node_name=self.name)
        return ml.make_cfg(raw_cfg, LenRabiCfgTemplate)

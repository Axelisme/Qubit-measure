"""lenrabi — length-Rabi Builder: acquire Rabi oscillation → fit_rabi → pi/pi2 lengths.

Translates the notebook's LenRabiTask cfg_maker. Sets this flux point's value on
the picked flux device, sets up devices, acquires a Rabi oscillation vs pulse
length with ``ModularProgramV2`` (Reset → rabi_pulse → Readout), fits it with the
real ``fit_rabi``, fills its sweep Result row in place, and returns the raw pi
and pi2 lengths plus the Rabi frequency.

- requires ``qubit_freq`` (a hard require via Dependency): the Rabi experiment
  drives the qubit on resonance, so no qubit frequency → no sensible cfg.
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).
- provides the ``pi_pulse`` and ``pi2_pulse`` modules built from the fitted pi /
  pi2 lengths.

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
from collections.abc import MutableMapping
from typing import Any, cast

import numpy as np

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2_gui.adapters.twotone.rabi.len_rabi import LenRabiAdapter
from zcu_tools.gui.app.autofluxdep.cfg import FloatSpec, SweepValue, str_choice_spec
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema, sweepcfg_to_axis
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    SnrProbe,
    acquire_to_complex,
    axis_to_sweep,
    build_stop_checkers,
    fill_decay_fit_or_skip,
    make_on_round,
    require_flux_device,
    round_progress,
    set_flux_by_name,
    signal2real_flip,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.defaults import (
    adapter_node_schema,
    ctx_md_float,
    ctx_module,
    generation_field,
    move_module,
    pulse_length,
    pulse_product,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import READOUT_LIBRARY_ALIASES
from zcu_tools.gui.app.autofluxdep.nodes.plotters import ColormapLinePlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
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
_DEFAULT_PI_PRODUCT_FACTOR = 1.5
_DEFAULT_MAX_DRIVE_GAIN = 1.0
_SWEEP_RANGE_MODE_AUTO_PI_LENGTH = "auto_pi_length"
_SWEEP_RANGE_MODE_FIXED = "fixed"
_RELAX_DELAY_MODE_AUTO_T1 = "auto_t1"
_RELAX_DELAY_MODE_FIXED = "fixed"
_DRIVE_GAIN_MODE_AUTO_PI_PRODUCT = "auto_pi_product"
_DRIVE_GAIN_MODE_FIXED = "fixed"


class LenRabiModuleCfg(ConfigBase):
    """The module bundle lenrabi lowers a context into (mirrors the lower-layer
    ``experiment/v2/autofluxdep`` LenRabiModuleCfg): an optional reset, the
    on-resonance ``rabi_pulse`` (the swept drive), and the ``readout``."""

    reset: ResetCfg | None = None
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


def _default_readout() -> Any | None:
    return None


def _default_none() -> None:
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _seed_t1(ctx: Any | None) -> float:
    return ctx_md_float(ctx, "t1") or _DEFAULT_T1


def _seed_pi_length(ctx: Any | None) -> float:
    md_value = ctx_md_float(ctx, "pi_len")
    if md_value is not None:
        return md_value
    module = ctx_module(ctx, "pi_amp", "pi_len", "pi_pulse")
    return pulse_length(module) or _DEFAULT_EXPECTED_PI_LENGTH


def _seed_pi_product(ctx: Any | None) -> float:
    module = ctx_module(ctx, "pi_amp", "pi_len", "pi_pulse")
    return pulse_product(module) or _seed_pi_length(ctx)


def _auto_relax_delay(t1: float, *, factor: float, minimum: float) -> float:
    return max(float(minimum), float(factor) * float(t1))


def _auto_sweep_range(
    pi_length: float, *, start: float, stop_factor: float, stop_min: float
) -> tuple[float, float]:
    return (float(start), max(float(stop_min), float(stop_factor) * float(pi_length)))


def _fixed_sweep_range(sweep: Any) -> tuple[float, float]:
    return (float(sweep.start), float(sweep.stop))


def _resolve_cfg_sweep_range(
    mode: str, *, pi_length: float, fixed: Any, knobs: dict[str, Any]
) -> tuple[float, float]:
    if mode == _SWEEP_RANGE_MODE_AUTO_PI_LENGTH:
        return _auto_sweep_range(
            pi_length,
            start=float(knobs["sweep_start_us"]),
            stop_factor=float(knobs["sweep_stop_factor"]),
            stop_min=float(knobs["sweep_stop_min_us"]),
        )
    if mode == _SWEEP_RANGE_MODE_FIXED:
        return _fixed_sweep_range(fixed)
    raise RuntimeError(f"unsupported lenrabi sweep_range_mode: {mode!r}")


def _resolve_cfg_relax_delay(
    mode: str, *, t1: float, fixed: float, knobs: dict[str, Any]
) -> float:
    if mode == _RELAX_DELAY_MODE_AUTO_T1:
        return _auto_relax_delay(
            t1,
            factor=float(knobs["relax_factor"]),
            minimum=float(knobs["relax_min_us"]),
        )
    if mode == _RELAX_DELAY_MODE_FIXED:
        return float(fixed)
    raise RuntimeError(f"unsupported lenrabi relax_delay_mode: {mode!r}")


def _drive_gain_from_pi_product(
    pi_product: float, pi_length: float, *, factor: float, max_drive_gain: float
) -> float:
    if pi_product <= 0.0:
        raise RuntimeError("lenrabi auto drive gain needs positive pi_product")
    if pi_length <= 0.0:
        raise RuntimeError("lenrabi auto drive gain needs positive pi_length")
    if factor <= 0.0:
        raise RuntimeError("lenrabi pi_product_factor must be positive")
    if max_drive_gain <= 0.0:
        raise RuntimeError("lenrabi max_drive_gain must be positive")
    return min(float(max_drive_gain), float(pi_product) / (factor * pi_length))


def _resolve_drive_gain(
    mode: str,
    *,
    pi_product: float,
    pi_length: float,
    fixed: float,
    knobs: dict[str, Any],
) -> float:
    if mode == _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT:
        return _drive_gain_from_pi_product(
            pi_product,
            pi_length,
            factor=float(knobs["pi_product_factor"]),
            max_drive_gain=float(knobs["max_drive_gain"]),
        )
    if mode == _DRIVE_GAIN_MODE_FIXED:
        return float(fixed)
    raise RuntimeError(f"unsupported lenrabi drive_gain_mode: {mode!r}")


def _drive_pulse_with_length(base: PulseCfg, length: float) -> dict[str, Any]:
    pulse = base.model_copy(deep=True)
    pulse.set_param("length", float(length))
    return pulse.to_dict()


class LenRabiNode(Node):
    """One flux point's lenrabi: set flux → real acquire → fit_rabi → fill row → Patch.

    Mirrors the lower-layer LenRabi Schedule acquire + ``run``: the on-resonance
    drive sweeps its pulse length, ``ModularProgramV2`` (Reset → rabi_pulse → Readout)
    acquires per round, and ``fit_rabi`` recovers the pi / pi2 lengths + Rabi freq.
    """

    def __init__(self, env: RunEnv, builder: LenRabiBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot["qubit_freq"]  # required — the on-resonance drive frequency
        _ = snapshot.module("opt_readout")  # optional — readout for the cfg path

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
        stop_checkers = build_stop_checkers(env, probe, signal2real_flip)
        with round_progress(cfg.rounds, "lenrabi", idx) as update_round_progress:
            on_round = make_on_round(
                result,
                idx,
                signal2real_flip,
                env.round_hook,
                probe=probe,
                round_progress_hook=update_round_progress,
            )
            raw = ModularProgramV2(
                env.soccfg,
                cfg,
                modules=[
                    Reset("reset", cfg.modules.reset),
                    Pulse("rabi_pulse", cfg.modules.rabi_pulse),
                    Readout("readout", cfg.modules.readout),
                ],
                sweep=[("length", length_sweep)],
            ).acquire(
                env.soc,
                progress=False,
                round_hook=on_round,
                stop_checkers=stop_checkers,
            )
        real = signal2real_flip(acquire_to_complex(raw))

        pi_x, _, pi2_x, _, freq, _, fit_curve, _ = fit_rabi(lengths, real)

        # The fitted single scalar (the Result's fit_value) is the pi length; the
        # extra Patch keys/modules below are lenrabi-specific and stay in the node.
        if not fill_decay_fit_or_skip(
            result, idx, real, float(pi_x), fit_curve, env.round_hook, logger, "lenrabi"
        ):
            # partial: omit pi_length/pi2_length/rabi_freq + modules → downstream fallback
            return Patch()

        logger.debug(
            "lenrabi fit @flux%d: rabi_freq=%.4f pi_len=%.3f pi2_len=%.3f",
            idx,
            float(freq),
            float(pi_x),
            float(pi2_x),
        )

        patch = Patch()
        patch.set("pi_length", float(pi_x))
        patch.set("pi2_length", float(pi2_x))
        patch.set("rabi_freq", float(freq))
        patch.set("pi_product", float(pi_x) * float(base_drive_pulse.gain))
        patch.set_module("pi_pulse", _drive_pulse_with_length(base_drive_pulse, pi_x))
        patch.set_module("pi2_pulse", _drive_pulse_with_length(base_drive_pulse, pi2_x))
        return patch


class LenRabiBuilder(Builder):
    """The lenrabi provider — acquire Rabi oscillation, real fit_rabi, accumulating
    colormap.  Produces pi_pulse and pi2_pulse modules in addition to the scalar
    pi_length / pi2_length / rabi_freq info values.
    """

    name = "lenrabi"
    provides = ("pi_length", "pi2_length", "rabi_freq", "pi_product")
    provides_modules = ("pi_pulse", "pi2_pulse")
    requires = (Dependency("qubit_freq"),)
    optional = (
        Dependency("t1", smooth="ewma", default=_default_none),
        Dependency("pi_length", default=_default_none),
        Dependency("pi_product", smooth="step_weighted", default=_default_none),
    )
    optional_modules = (
        ModuleDep(
            "opt_readout", default=_default_readout, aliases=READOUT_LIBRARY_ALIASES
        ),
    )

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Adapter-backed default cfg plus autofluxdep generation controls."""
        t1_seed = _seed_t1(ctx)
        pi_len_seed = _seed_pi_length(ctx)
        return adapter_node_schema(
            LenRabiAdapter,
            ctx,
            logical_paths={
                "reset": "modules.reset",
                "rabi_pulse": "modules.qub_pulse",
                "qub_ch": "modules.qub_pulse.ch",
                "qub_nqz": "modules.qub_pulse.nqz",
                "qub_gain": "modules.qub_pulse.gain",
                "readout": "modules.readout",
                "relax_delay": "relax_delay",
                "reps": "reps",
                "rounds": "rounds",
                "sweep_range": "sweep.length",
            },
            generation_fields=(
                generation_field(
                    "earlystop_snr",
                    "earlystop_snr",
                    FloatSpec(label="earlystop_snr", optional=True),
                    _DEFAULT_EARLYSTOP_SNR,
                    group="safety",
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
                    "relax_min_us",
                    "relax_min_us",
                    FloatSpec(label="relax_min_us"),
                    _DEFAULT_RELAX_MIN,
                    group="timing",
                ),
                generation_field(
                    "sweep_range_mode",
                    "sweep_range_mode",
                    str_choice_spec(
                        "sweep_range_mode",
                        (_SWEEP_RANGE_MODE_AUTO_PI_LENGTH, _SWEEP_RANGE_MODE_FIXED),
                    ),
                    _SWEEP_RANGE_MODE_AUTO_PI_LENGTH,
                    group="sweep",
                ),
                generation_field(
                    "expected_pi_length",
                    "expected_pi_length",
                    FloatSpec(label="expected_pi_length"),
                    pi_len_seed,
                    group="sweep",
                ),
                generation_field(
                    "sweep_start_us",
                    "sweep_start_us",
                    FloatSpec(label="sweep_start_us"),
                    _DEFAULT_SWEEP_START,
                    group="sweep",
                ),
                generation_field(
                    "sweep_stop_factor",
                    "sweep_stop_factor",
                    FloatSpec(label="sweep_stop_factor"),
                    _DEFAULT_SWEEP_STOP_FACTOR,
                    group="sweep",
                ),
                generation_field(
                    "sweep_stop_min_us",
                    "sweep_stop_min_us",
                    FloatSpec(label="sweep_stop_min_us"),
                    _DEFAULT_SWEEP_STOP_MIN,
                    group="sweep",
                ),
                generation_field(
                    "drive_gain_mode",
                    "drive_gain_mode",
                    str_choice_spec(
                        "drive_gain_mode",
                        (_DRIVE_GAIN_MODE_AUTO_PI_PRODUCT, _DRIVE_GAIN_MODE_FIXED),
                    ),
                    _DRIVE_GAIN_MODE_AUTO_PI_PRODUCT,
                    group="feedback",
                ),
                generation_field(
                    "pi_product_seed",
                    "pi_product_seed",
                    FloatSpec(label="pi_product_seed"),
                    _seed_pi_product(ctx),
                    group="feedback",
                ),
                generation_field(
                    "pi_product_factor",
                    "pi_product_factor",
                    FloatSpec(label="pi_product_factor"),
                    _DEFAULT_PI_PRODUCT_FACTOR,
                    group="feedback",
                ),
                generation_field(
                    "max_drive_gain",
                    "max_drive_gain",
                    FloatSpec(label="max_drive_gain"),
                    _DEFAULT_MAX_DRIVE_GAIN,
                    group="feedback",
                ),
            ),
            default_overrides={
                "rounds": 10,
                "relax_delay": _auto_relax_delay(
                    t1_seed,
                    factor=_DEFAULT_RELAX_FACTOR,
                    minimum=_DEFAULT_RELAX_MIN,
                ),
                "sweep_range": SweepValue(
                    *_auto_sweep_range(
                        pi_len_seed,
                        start=_DEFAULT_SWEEP_START,
                        stop_factor=_DEFAULT_SWEEP_STOP_FACTOR,
                        stop_min=_DEFAULT_SWEEP_STOP_MIN,
                    ),
                    expts=101,
                ),
            },
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
        raw_cfg = env.schema.lower_raw(ml, md=env.md)
        knobs = env.schema.lower(ml, md=env.md)
        qubit_freq = float(snapshot["qubit_freq"])
        t1 = _float_or_none(snapshot.get("t1")) or float(knobs["t1_seed_us"])
        prev_pi_length = _float_or_none(snapshot.get("pi_length")) or float(
            knobs["expected_pi_length"]
        )
        prev_pi_product = _float_or_none(snapshot.get("pi_product")) or float(
            knobs["pi_product_seed"]
        )

        move_module(raw_cfg, "qub_pulse", "rabi_pulse")
        raw_cfg["modules"]["rabi_pulse"]["gain"] = _resolve_drive_gain(
            str(knobs["drive_gain_mode"]),
            pi_product=prev_pi_product,
            pi_length=prev_pi_length,
            fixed=float(raw_cfg["modules"]["rabi_pulse"]["gain"]),
            knobs=knobs,
        )
        raw_cfg["modules"]["rabi_pulse"]["freq"] = qubit_freq
        raw_cfg["modules"]["readout"] = readout
        raw_cfg.pop("sweep", None)
        raw_cfg["relax_delay"] = _resolve_cfg_relax_delay(
            str(knobs["relax_delay_mode"]),
            t1=t1,
            fixed=float(knobs["relax_delay"]),
            knobs=knobs,
        )
        raw_cfg["sweep_range"] = _resolve_cfg_sweep_range(
            str(knobs["sweep_range_mode"]),
            pi_length=prev_pi_length,
            fixed=knobs["sweep_range"],
            knobs=knobs,
        )
        return ml.make_cfg(raw_cfg, LenRabiCfgTemplate)

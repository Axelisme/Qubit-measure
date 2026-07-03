"""t2echo — Hahn-echo Builder: acquire decay/fringe trace → fit → t2e.

Translates the notebook's T2EchoTask cfg_maker. Sets this flux point's value on
the picked flux device, sets up devices, acquires a decaying cosine fringe vs
delay time with ``ModularProgramV2`` (a Hahn-echo sequence), fits it with either
``fit_decay_fringe`` or ``fit_decay`` according to the generation fit method, fills
its sweep Result row in place, and returns the raw t2e.

Unlike t2ramsey, the echo sequence refocuses static dephasing and typically
yields a longer coherence time; the difference is purely in the pulse sequence.
The default ``auto_by_detune`` fit method uses a pure decay fit when
``detune_ratio == 0`` and a fringe fit otherwise.

- needs the ``pi_pulse`` and ``pi2_pulse`` modules (lenrabi produces both) — the
  Hahn echo needs both a pi refocusing pulse and two pi/2 pulses. Both carry
  placeholder defaults, so they never actually skip when lenrabi is absent.
- reads ``t1`` (smooth="ewma") and ``t2e`` (smooth="ewma") as optional deps:
  ``t2e`` seeds the planted t2 so the sweep tracks a plausible echo time;
  ``t1`` is available for cfg sanity checks (not used directly in the prototype).
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).

``produce`` lowers the active context (a populated ``ml`` + the upstream
``pi_pulse`` / ``pi2_pulse`` / ``opt_readout`` modules on the snapshot, real
``PulseCfg`` / ``ReadoutCfg`` lenrabi/ro_optimize output) into a runnable
``T2EchoCfgTemplate`` via ``ml.make_cfg`` (mirroring the notebook's T2EchoTask
cfg_maker), takes the delay-time window (``sweep_range``) from the built cfg, and
acquires against a flux-aware MockSoc (offline) or real hardware. The cfg is the
source of the measurement window; ``make_cfg`` Fast Fails when the context is
unconfigured.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any, cast

import numpy as np

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2_gui.adapters.twotone.time_domain.t2echo import (
    T2EchoAdapter,
)
from zcu_tools.gui.app.autofluxdep.cfg import FloatSpec, SweepValue, str_choice_spec
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema, sweepcfg_to_axis
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    SnrProbe,
    acquire_to_complex,
    axis_to_sweep,
    build_stop_checkers,
    is_good_fit,
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
    generation_field,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import (
    PI2_PULSE_LIBRARY_ALIASES,
    PI_PULSE_LIBRARY_ALIASES,
    READOUT_LIBRARY_ALIASES,
)
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Decay1DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    Readout,
    Reset,
    sweep2param,
)
from zcu_tools.program.v2.modules import PulseCfg, ReadoutCfg, ResetCfg
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe

logger = logging.getLogger(__name__)

_DEFAULT_T1 = 10.0  # us — smoothed t1 fallback
_DEFAULT_T2E = 5.0  # us — smoothed t2e fallback
_T2E_WINDOW_FACTOR = 2.5  # notebook: sweep_range = (0, 2.5 * prev_t2e)
_DEFAULT_DETUNE_RATIO = 0.05  # notebook default activate-detune fraction
_DEFAULT_EARLYSTOP_SNR = 20.0
_DEFAULT_RELAX_FACTOR = 3.0
_DEFAULT_RELAX_MIN = 1.0
_DEFAULT_SWEEP_START = 0.0
_SWEEP_RANGE_MODE_AUTO_T2E = "auto_t2e"
_SWEEP_RANGE_MODE_FIXED = "fixed"
_RELAX_DELAY_MODE_AUTO_T1 = "auto_t1"
_RELAX_DELAY_MODE_FIXED = "fixed"
_FIT_METHOD_AUTO = "auto_by_detune"
_FIT_METHOD_FRINGE = "fringe"
_FIT_METHOD_DECAY = "decay"


def _default_t1() -> None:
    return None


def _default_t2e() -> None:
    return None


def _seed_t1(ctx: Any | None) -> float:
    return ctx_md_float(ctx, "t1") or _DEFAULT_T1


def _seed_t2e(ctx: Any | None) -> float:
    return ctx_md_float(ctx, "t2e") or _DEFAULT_T2E


def _snapshot_float(snapshot: Snapshot, key: str, fallback: float) -> float:
    value = snapshot.get(key)
    if value is None:
        return fallback
    return float(value)


def _placeholder_pi_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real (placeholder) module
    return {"type": "pi", "length": 0.1}


def _placeholder_pi2_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real (placeholder) module
    return {"type": "pi2", "length": 0.05}


def _default_readout() -> Any | None:
    return None


def _fixed_sweep_range(sweep: Any) -> tuple[float, float]:
    return (float(sweep.start), float(sweep.stop))


def _resolve_cfg_sweep_range(
    mode: str, *, t2e: float, fixed: Any, knobs: dict[str, Any]
) -> tuple[float, float]:
    if mode == _SWEEP_RANGE_MODE_AUTO_T2E:
        return (
            float(knobs["sweep_start_us"]),
            float(knobs["sweep_stop_factor"]) * float(t2e),
        )
    if mode == _SWEEP_RANGE_MODE_FIXED:
        return _fixed_sweep_range(fixed)
    raise RuntimeError(f"unsupported t2echo sweep_range_mode: {mode!r}")


def _resolve_cfg_relax_delay(
    mode: str, *, t1: float, fixed: float, knobs: dict[str, Any]
) -> float:
    if mode == _RELAX_DELAY_MODE_AUTO_T1:
        return max(float(knobs["relax_min_us"]), float(knobs["relax_factor"]) * t1)
    if mode == _RELAX_DELAY_MODE_FIXED:
        return float(fixed)
    raise RuntimeError(f"unsupported t2echo relax_delay_mode: {mode!r}")


def _is_lowerable_pulse(module: Any) -> bool:
    """Whether a resolved drive module is a concrete, lowerable ``PulseCfg``.

    A real lenrabi drive pulse is a ``PulseCfg`` (or its raw dict, ``type ==
    "pulse"``) and lowers into the run cfg. The prototype's placeholder
    ``{"type": "pi"/"pi2", "length": ...}`` is NOT a PulseCfg (it never
    validates), so this returns False there — the guard then rejects the
    placeholder so ``make_cfg`` Fast Fails for an unconfigured context. Mirrors
    qubit_freq's guard naturally returning None in that context.
    """
    if isinstance(module, PulseCfg):
        return True
    if isinstance(module, dict):
        return module.get("type") == "pulse"
    return False


def _fit_t2echo(
    method: str, *, detune_ratio: float, times: np.ndarray, real: np.ndarray
) -> tuple[float, Any]:
    if method == _FIT_METHOD_AUTO:
        method = _FIT_METHOD_DECAY if detune_ratio == 0.0 else _FIT_METHOD_FRINGE
    if method == _FIT_METHOD_DECAY:
        t2f, _, fit_curve, _ = fit_decay(times, real)
        return float(t2f), fit_curve
    if method == _FIT_METHOD_FRINGE:
        t2f, _, _, _, fit_curve, _ = fit_decay_fringe(times, real)
        return float(t2f), fit_curve
    raise RuntimeError(f"unsupported t2echo fit_method: {method!r}")


class T2EchoModuleCfg(ConfigBase):
    """The module bundle a t2echo run cfg carries.

    Mirrors the lower-layer ``experiment/v2/autofluxdep`` ``T2EchoModuleCfg``: an
    optional reset, the pi refocusing pulse, the pi/2 pulse (used twice in the
    Hahn-echo sequence), and the readout. ``pi_pulse`` / ``pi2_pulse`` are the
    lenrabi-produced drive pulses; ``readout`` is the (optionally optimised)
    readout module.
    """

    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class T2EchoCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base Hahn-echo cfg t2echo lowers a context into.

    ``ProgramV2Cfg`` (reps/rounds/relax) + the ``ExpCfgModel`` device/save fields
    + the t2echo modules and the ``sweep_range`` delay window — same bases as the
    lower-layer ``experiment/v2/autofluxdep`` ``T2EchoCfgTemplate``. The flux
    ``dev`` entry and the concrete ``length`` sweep are merged in by ``produce``;
    here ``produce`` reads the ``sweep_range`` window to parameterise the acquire.
    """

    modules: T2EchoModuleCfg
    sweep_range: tuple[float, float]


class T2EchoNode(Node):
    """One flux point's t2echo: set flux → real acquire → configured fit → Patch.

    Mirrors the lower-layer T2Echo Schedule acquire + ``run``: a
    Hahn-echo sequence (pi/2 → τ/2 → pi → τ/2 → optional detuned pi/2) sweeps the
    total delay τ, and the configured fit method recovers T2Echo.
    """

    def __init__(self, env: RunEnv, builder: T2EchoBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot.module("pi_pulse")  # required — refocusing pulse
        _ = snapshot.module("pi2_pulse")  # required — the two pi/2 pulses
        _ = snapshot.module("opt_readout")  # required — readout

        result: Sweep1DResult = env.result
        idx = env.flux_idx

        # Lower the active context into the run cfg (Fast Fail if unconfigured: a
        # real acquire needs concrete pi / pi2 drive pulses + a readout). The cfg's
        # sweep_range = (0, 2.5 × smoothed_t2e) sets the total-delay axis.
        cfg = self._builder.make_cfg(env, snapshot)
        lo, hi = float(cfg.sweep_range[0]), float(cfg.sweep_range[1])
        times = np.linspace(lo, hi, result.n_x)
        result.x[:] = times

        flux_device = require_flux_device(env, "t2echo")
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

        # The total-delay sweep, split across the two echo halves (Delay 0.5·τ each),
        # + the activate-detune phase ramp on the 2nd pi/2 (lower layer:
        # activate_detune = detune_ratio / len_sweep.step).
        length_sweep = axis_to_sweep(times)
        length_param = sweep2param("length", length_sweep)
        detune_ratio = self._builder.detune_ratio(env.schema, md=env.md)
        activate_detune = detune_ratio / length_sweep.step
        pi2_pulse = cfg.modules.pi2_pulse

        result.flux[idx] = env.flux

        probe = SnrProbe()
        stop_checkers = build_stop_checkers(env, probe, signal2real_flip)
        with round_progress(cfg.rounds, "t2echo", idx) as update_round_progress:
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
                    Pulse("pi2_pulse1", pi2_pulse),
                    Delay("t2e_delay1", delay=0.5 * length_param),
                    Pulse("pi_pulse", cfg.modules.pi_pulse),
                    Delay("t2e_delay2", delay=0.5 * length_param),
                    Pulse(
                        "pi2_pulse2",
                        pi2_pulse.with_updates(
                            phase=pi2_pulse.phase + 360 * activate_detune * length_param
                        ),
                    ),
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

        fit_method = self._builder.fit_method(env.schema, md=env.md)
        t2f, fit_curve = _fit_t2echo(
            fit_method, detune_ratio=detune_ratio, times=times, real=real
        )

        if not is_good_fit(real, fit_curve):
            logger.debug("t2echo fit @flux%d: poor fit (SNR-trough?) — discarded", idx)
            if env.round_hook is not None:
                env.round_hook(idx)  # raw row already shown; fit fields stay nan
            return Patch()  # partial: omit t2e → downstream fallback

        result.fit_value[idx] = float(t2f)
        np.copyto(result.fit_curve[idx], np.asarray(fit_curve, dtype=np.float64))
        if env.round_hook is not None:
            env.round_hook(idx)

        logger.debug("t2echo fit @flux%d: t2e=%.3f us", idx, float(t2f))

        patch = Patch()
        patch.set("t2e", float(t2f))
        return patch


class T2EchoBuilder(Builder):
    """The t2echo provider — acquire echo decay/fringe traces and fit T2Echo.

    Reports only the raw echo t2e (detune is refocused and not reported).
    """

    name = "t2echo"
    provides = ("t2e",)
    optional = (
        Dependency("t1", smooth="ewma", default=_default_t1),
        Dependency("t2e", smooth="ewma", default=_default_t2e),
    )
    requires_modules = (
        ModuleDep(
            "pi_pulse", default=_placeholder_pi_pulse, aliases=PI_PULSE_LIBRARY_ALIASES
        ),
        ModuleDep(
            "pi2_pulse",
            default=_placeholder_pi2_pulse,
            aliases=PI2_PULSE_LIBRARY_ALIASES,
        ),
    )
    optional_modules = (
        ModuleDep(
            "opt_readout", default=_default_readout, aliases=READOUT_LIBRARY_ALIASES
        ),
    )

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Adapter-backed default cfg plus autofluxdep generation controls."""
        t1_seed = _seed_t1(ctx)
        t2e_seed = _seed_t2e(ctx)
        return adapter_node_schema(
            T2EchoAdapter,
            ctx,
            logical_paths={
                "reset": "modules.reset",
                "pi_pulse": "modules.pi_pulse",
                "pi2_pulse": "modules.pi2_pulse",
                "readout": "modules.readout",
                "relax_delay": "relax_delay",
                "detune_ratio": "detune_ratio",
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
                ),
                generation_field(
                    "sweep_range_mode",
                    "sweep_range_mode",
                    str_choice_spec(
                        "sweep_range_mode",
                        (_SWEEP_RANGE_MODE_AUTO_T2E, _SWEEP_RANGE_MODE_FIXED),
                    ),
                    _SWEEP_RANGE_MODE_AUTO_T2E,
                ),
                generation_field(
                    "relax_delay_mode",
                    "relax_delay_mode",
                    str_choice_spec(
                        "relax_delay_mode",
                        (_RELAX_DELAY_MODE_AUTO_T1, _RELAX_DELAY_MODE_FIXED),
                    ),
                    _RELAX_DELAY_MODE_AUTO_T1,
                ),
                generation_field(
                    "t1_seed_us",
                    "t1_seed_us",
                    FloatSpec(label="t1_seed_us"),
                    t1_seed,
                ),
                generation_field(
                    "t2e_seed_us",
                    "t2e_seed_us",
                    FloatSpec(label="t2e_seed_us"),
                    t2e_seed,
                ),
                generation_field(
                    "relax_factor",
                    "relax_factor",
                    FloatSpec(label="relax_factor"),
                    _DEFAULT_RELAX_FACTOR,
                ),
                generation_field(
                    "relax_min_us",
                    "relax_min_us",
                    FloatSpec(label="relax_min_us"),
                    _DEFAULT_RELAX_MIN,
                ),
                generation_field(
                    "sweep_start_us",
                    "sweep_start_us",
                    FloatSpec(label="sweep_start_us"),
                    _DEFAULT_SWEEP_START,
                ),
                generation_field(
                    "sweep_stop_factor",
                    "sweep_stop_factor",
                    FloatSpec(label="sweep_stop_factor"),
                    _T2E_WINDOW_FACTOR,
                ),
                generation_field(
                    "fit_method",
                    "fit_method",
                    str_choice_spec(
                        "fit_method",
                        (_FIT_METHOD_AUTO, _FIT_METHOD_FRINGE, _FIT_METHOD_DECAY),
                    ),
                    _FIT_METHOD_AUTO,
                ),
            ),
            default_overrides={
                "detune_ratio": _DEFAULT_DETUNE_RATIO,
                "rounds": 10,
                "relax_delay": max(_DEFAULT_RELAX_MIN, _DEFAULT_RELAX_FACTOR * t1_seed),
                "sweep_range": SweepValue(
                    _DEFAULT_SWEEP_START,
                    _T2E_WINDOW_FACTOR * t2e_seed,
                    expts=101,
                ),
            },
        )

    def detune_ratio(self, schema: NodeCfgSchema, md: Any = None) -> float:
        """The activate-detune ratio for this placement (typed knob, default 0.05)."""
        return float(schema.lower(None, md=md)["detune_ratio"])

    def fit_method(self, schema: NodeCfgSchema, md: Any = None) -> str:
        return str(schema.lower(None, md=md)["fit_method"])

    def make_init_result(
        self, schema: NodeCfgSchema, flux: Any, md: Any = None
    ) -> Sweep1DResult:
        knobs = schema.lower(None, md=md)
        times = sweepcfg_to_axis(knobs["sweep_range"])
        return Sweep1DResult.allocate(flux, times, x_label="delay time (us)")

    def make_plotter(self, figure: Any) -> Decay1DPlotter:
        return Decay1DPlotter(
            figure, title="t2echo", value_label="T2 Echo (us)", x_label="Time (us)"
        )

    def build_node(self, env: RunEnv) -> T2EchoNode:
        return T2EchoNode(env, self)

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> T2EchoCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the notebook's t2echo ``cfg_maker``: the pi / pi2 drive pulses are
        the latest-available lenrabi-produced ``pi_pulse`` / ``pi2_pulse`` modules
        on the snapshot, the readout is the latest-available ``opt_readout``
        module, the relax delay is ``max(1.0, 3 * smoothed_t1)``, and the
        ``sweep_range`` delay window is ``(0, 2.5 * smoothed_t2e)``. The flux
        ``dev`` entry and the concrete ``length`` sweep are NOT here — the
        lower-layer ``run`` merges them.

        Raises if the ml / drive pulses / readout are unavailable — a real run
        needs concrete drive pulses (Fast Fail).
        """
        ml = env.ml
        if ml is None:
            raise RuntimeError("t2echo.make_cfg needs an active ModuleLibrary")
        pi_pulse = snapshot.module("pi_pulse")
        pi2_pulse = snapshot.module("pi2_pulse")
        readout = snapshot.module("opt_readout")
        if not _is_lowerable_pulse(pi_pulse) or not _is_lowerable_pulse(pi2_pulse):
            raise RuntimeError(
                "t2echo.make_cfg needs concrete pi_pulse / pi2_pulse drive modules "
                "(lenrabi output)"
            )
        if readout is None:
            raise RuntimeError(
                "t2echo.make_cfg needs a readout module (none produced or preset)"
            )
        raw_cfg = env.schema.lower_raw(ml, md=env.md)
        knobs = env.schema.lower(ml, md=env.md)
        cur_t1 = _snapshot_float(snapshot, "t1", float(knobs["t1_seed_us"]))
        prev_t2e = _snapshot_float(snapshot, "t2e", float(knobs["t2e_seed_us"]))
        relax_delay = _resolve_cfg_relax_delay(
            str(knobs["relax_delay_mode"]),
            t1=cur_t1,
            fixed=float(knobs["relax_delay"]),
            knobs=knobs,
        )
        sweep_range = _resolve_cfg_sweep_range(
            str(knobs["sweep_range_mode"]),
            t2e=prev_t2e,
            fixed=knobs["sweep_range"],
            knobs=knobs,
        )
        raw_cfg["modules"]["pi_pulse"] = pi_pulse
        raw_cfg["modules"]["pi2_pulse"] = pi2_pulse
        raw_cfg["modules"]["readout"] = readout
        raw_cfg.pop("sweep", None)
        raw_cfg.pop("detune_ratio", None)
        raw_cfg["relax_delay"] = relax_delay
        raw_cfg["sweep_range"] = sweep_range
        return ml.make_cfg(raw_cfg, T2EchoCfgTemplate)

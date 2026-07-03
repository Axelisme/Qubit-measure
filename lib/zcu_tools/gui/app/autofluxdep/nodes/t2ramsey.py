"""t2ramsey — Ramsey-fringe Builder: acquire decay cosine → fit_decay_fringe → t2r.

Translates the notebook's T2RamseyTask cfg_maker. Sets this flux point's value on
the picked flux device, sets up devices, acquires a decaying cosine fringe vs
delay time with ``ModularProgramV2`` (two pi/2 pulses bracketing a swept delay,
the second carrying an activate-detune phase ramp), fits it with the real
``fit_decay_fringe``, fills its sweep Result row in place, and returns the raw t2r
and the measured detune.

- requires the ``pi2_pulse`` module (lenrabi produces it) — the Ramsey sequence
  needs a pi/2 pulse; it is a required module dep with a placeholder default for
  the prototype.
- reads ``t1`` (smooth="ewma") and ``t2r`` (smooth="ewma") as optional deps:
  ``t2r`` seeds the planted t2 so the sweep tracks a plausible decoherence time;
  ``t1`` is available for cfg sanity checks (not used directly in the prototype).
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).

``make_cfg`` lowers the active context (a populated ml + the ``pi2_pulse`` drive
module + the ``opt_readout`` readout module both present on the snapshot) + this
point's snapshot into the base ``T2RamseyCfgTemplate`` — exercising the real
``ml.make_cfg`` pipeline — and ``produce`` takes the delay-time window from the
cfg's ``sweep_range`` (which encodes ``2.5 * prev_t2r``), then acquires against a
flux-aware MockSoc (offline) or real hardware. ``make_cfg`` Fast Fails when the
context is unconfigured.

Compare ``notebook_md/autofluxdep.md`` (the T2RamseyTask block):

    cfg_maker=lambda ctx, ml: (
        (info := ctx.env["info"])
        and (cur_t1 := info.get("smooth_t1", md.t1))                 # relax_delay
        and (prev_t2r := info.last.get("smooth_t2r", md.t2r))        # sweep_range
        and (cur_pi2_pulse := info.get("pi2_pulse"))                 # required module
        and (opt_readout := info.last.get("opt_readout", readout_cfg))  # optional
        and ml.make_cfg({"modules": {"pi2_pulse": cur_pi2_pulse,
                                     "readout": opt_readout},
                         "relax_delay": max(1.0, 3 * cur_t1),
                         "reps": 1000, "rounds": 10,
                         "sweep_range": (0, 2.5 * prev_t2r)}, ...) )
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any, cast

import numpy as np

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.autofluxdep.t2ramsey import T2RamseyModuleCfg
from zcu_tools.experiment.v2_gui.adapters.twotone.time_domain.t2ramsey import (
    T2RamseyAdapter,
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
from zcu_tools.utils.fitting import fit_decay_fringe

logger = logging.getLogger(__name__)

_DEFAULT_T1 = 10.0  # us — smoothed t1 fallback
_DEFAULT_T2R = 5.0  # us — smoothed t2r fallback
_SWEEP_T2R_FACTOR = 2.5  # notebook: sweep_range = (0, 2.5 * prev_t2r)
_DEFAULT_DETUNE_RATIO = 0.05  # notebook default activate-detune fraction
_DEFAULT_EARLYSTOP_SNR = 20.0
_DEFAULT_RELAX_FACTOR = 3.0
_DEFAULT_RELAX_MIN = 1.0
_DEFAULT_SWEEP_START = 0.0
_SWEEP_RANGE_MODE_AUTO_T2R = "auto_t2r"
_SWEEP_RANGE_MODE_FIXED = "fixed"
_RELAX_DELAY_MODE_AUTO_T1 = "auto_t1"
_RELAX_DELAY_MODE_FIXED = "fixed"


class T2RamseyCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base Ramsey cfg t2ramsey lowers a context into.

    ``ProgramV2Cfg`` (reps/rounds/relax) + the ``ExpCfgModel`` device/save fields,
    plus the Ramsey ``modules`` (``pi2_pulse`` + ``readout``, optional ``reset``)
    and a free ``sweep_range`` (the delay-time span) — mirroring the lower-layer
    ``experiment/v2/autofluxdep`` ``T2RamseyCfgTemplate``. The flux ``dev`` entry,
    the concrete ``length`` sweep, and ``activate_detune`` are merged in by the
    lower-layer ``run()`` (not here): this template is the cfg-maker output, and
    ``produce`` reads the planted-t2 baseline from ``sweep_range``.
    """

    modules: T2RamseyModuleCfg
    sweep_range: tuple[float, float]


def _default_t1() -> None:
    return None


def _default_t2r() -> None:
    return None


def _seed_t1(ctx: Any | None) -> float:
    return ctx_md_float(ctx, "t1") or _DEFAULT_T1


def _seed_t2r(ctx: Any | None) -> float:
    return ctx_md_float(ctx, "t2r") or _DEFAULT_T2R


def _snapshot_float(snapshot: Snapshot, key: str, fallback: float) -> float:
    value = snapshot.get(key)
    if value is None:
        return fallback
    return float(value)


def _placeholder_pi2_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real (placeholder) module
    return {"type": "pi2", "length": 0.05}


def _default_readout() -> Any | None:
    return None


def _fixed_sweep_range(sweep: Any) -> tuple[float, float]:
    return (float(sweep.start), float(sweep.stop))


def _resolve_cfg_sweep_range(
    mode: str, *, t2r: float, fixed: Any, knobs: dict[str, Any]
) -> tuple[float, float]:
    if mode == _SWEEP_RANGE_MODE_AUTO_T2R:
        return (
            float(knobs["sweep_start_us"]),
            float(knobs["sweep_stop_factor"]) * float(t2r),
        )
    if mode == _SWEEP_RANGE_MODE_FIXED:
        return _fixed_sweep_range(fixed)
    raise RuntimeError(f"unsupported t2ramsey sweep_range_mode: {mode!r}")


def _resolve_cfg_relax_delay(
    mode: str, *, t1: float, fixed: float, knobs: dict[str, Any]
) -> float:
    if mode == _RELAX_DELAY_MODE_AUTO_T1:
        return max(float(knobs["relax_min_us"]), float(knobs["relax_factor"]) * t1)
    if mode == _RELAX_DELAY_MODE_FIXED:
        return float(fixed)
    raise RuntimeError(f"unsupported t2ramsey relax_delay_mode: {mode!r}")


class T2RamseyNode(Node):
    """One flux point's t2ramsey: set flux → real acquire → fit_decay_fringe → Patch.

    Mirrors the lower-layer T2Ramsey Schedule acquire + ``run``: two
    pi/2 pulses bracket a swept delay, the second carries an activate-detune phase
    ramp (``360·detune·length``) so the fringe is resolvable, and
    ``fit_decay_fringe`` recovers T2Ramsey + the measured detune (the activate
    detune is subtracted back out, as the lower layer does).
    """

    def __init__(self, env: RunEnv, builder: T2RamseyBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot.module("pi2_pulse")  # required module — lowered into the cfg
        _ = snapshot.module("opt_readout")  # required — lowered into the cfg

        result: Sweep1DResult = env.result
        times = result.x
        idx = env.flux_idx

        # Lower the active context into the run cfg (Fast Fail if unconfigured: a
        # real acquire needs a concrete pi/2 pulse + readout). sweep_range encodes
        # 2.5 × smoothed_t2r; rebuild the delay axis over it.
        cfg = self._builder.make_cfg(env, snapshot)
        lo, hi = cfg.sweep_range
        times = np.linspace(float(lo), float(hi), result.n_x)
        result.x[:] = times

        flux_device = require_flux_device(env, "t2ramsey")
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

        # The Ramsey delay sweep + the activate-detune phase ramp on the 2nd pi/2
        # (lower layer: activate_detune = detune_ratio / len_sweep.step).
        length_sweep = axis_to_sweep(times)
        length_param = sweep2param("length", length_sweep)
        detune_ratio = self._builder.detune_ratio(env.schema, md=env.md)
        activate_detune = detune_ratio / length_sweep.step
        pi2_pulse = cfg.modules.pi2_pulse

        result.flux[idx] = env.flux

        probe = SnrProbe()
        stop_checkers = build_stop_checkers(env, probe, signal2real_flip)
        with round_progress(cfg.rounds, "t2ramsey", idx) as update_round_progress:
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
                    Delay("t2r_delay", delay=length_param),
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

        t2f, _, detune, _, fit_curve, _ = fit_decay_fringe(times, real)
        detune = detune - activate_detune  # back out the applied activate-detune

        if not is_good_fit(real, fit_curve):
            logger.debug(
                "t2ramsey fit @flux%d: poor fit (SNR-trough?) — discarded", idx
            )
            if env.round_hook is not None:
                env.round_hook(idx)  # raw row already shown; fit fields stay nan
            return Patch()  # partial: omit t2r/t2r_detune → downstream fallback

        result.fit_value[idx] = float(t2f)
        np.copyto(result.fit_curve[idx], np.asarray(fit_curve, dtype=np.float64))
        if env.round_hook is not None:
            env.round_hook(idx)

        logger.debug(
            "t2ramsey fit @flux%d: t2r=%.3f us detune=%.4f",
            idx,
            float(t2f),
            float(detune),
        )

        patch = Patch()
        patch.set("t2r", float(t2f))
        patch.set("t2r_detune", float(detune))
        return patch


class T2RamseyBuilder(Builder):
    """The t2ramsey provider — acquire decay cosine, real fit_decay_fringe, accumulating
    colormap.  Reports the raw Ramsey t2r and the measured detuning detune.
    """

    name = "t2ramsey"
    provides = ("t2r", "t2r_detune")
    optional = (
        Dependency("t1", smooth="ewma", default=_default_t1),
        Dependency("t2r", smooth="ewma", default=_default_t2r),
    )
    requires_modules = (
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
        t2r_seed = _seed_t2r(ctx)
        return adapter_node_schema(
            T2RamseyAdapter,
            ctx,
            logical_paths={
                "reset": "modules.reset",
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
                        (_SWEEP_RANGE_MODE_AUTO_T2R, _SWEEP_RANGE_MODE_FIXED),
                    ),
                    _SWEEP_RANGE_MODE_AUTO_T2R,
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
                    "t2r_seed_us",
                    "t2r_seed_us",
                    FloatSpec(label="t2r_seed_us"),
                    t2r_seed,
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
                    _SWEEP_T2R_FACTOR,
                ),
            ),
            default_overrides={
                "rounds": 10,
                "relax_delay": max(_DEFAULT_RELAX_MIN, _DEFAULT_RELAX_FACTOR * t1_seed),
                "sweep_range": SweepValue(
                    _DEFAULT_SWEEP_START,
                    _SWEEP_T2R_FACTOR * t2r_seed,
                    expts=101,
                ),
            },
        )

    def detune_ratio(self, schema: NodeCfgSchema, md: Any = None) -> float:
        """The activate-detune ratio for this placement (typed knob, default 0.05)."""
        return float(schema.lower(None, md=md)["detune_ratio"])

    def make_init_result(
        self, schema: NodeCfgSchema, flux: Any, md: Any = None
    ) -> Sweep1DResult:
        knobs = schema.lower(None, md=md)
        times = sweepcfg_to_axis(knobs["sweep_range"])
        return Sweep1DResult.allocate(flux, times, x_label="delay time (us)")

    def make_plotter(self, figure: Any) -> Decay1DPlotter:
        return Decay1DPlotter(
            figure, title="t2ramsey", value_label="T2Ramsey (us)", x_label="Time (us)"
        )

    def build_node(self, env: RunEnv) -> T2RamseyNode:
        return T2RamseyNode(env, self)

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> T2RamseyCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the notebook's t2ramsey ``cfg_maker`` (runs in ``produce``, where
        the snapshot is available): the ``pi2_pulse`` drive module and the
        ``readout`` module come from the snapshot (lenrabi / ro_optimize produce
        them, ml-preset / default otherwise), ``relax_delay`` is ``3 * t1`` (the
        smoothed t1 from the snapshot, floored at 1 us), the ``sweep_range`` spans
        ``2.5 * t2r`` (the smoothed t2r), and ``reps`` / ``rounds`` come from the
        node's params. The flux ``dev`` entry, the concrete ``length`` sweep, and
        ``activate_detune`` are NOT here — the lower-layer ``run()`` merges them.

        Raises if the ml / drive / readout modules are unavailable — a real run
        needs a concrete Ramsey sequence (Fast Fail).
        """
        ml = env.ml
        if ml is None:
            raise RuntimeError("t2ramsey.make_cfg needs an active ModuleLibrary")
        pi2_pulse = snapshot.module("pi2_pulse")
        if pi2_pulse is None:
            raise RuntimeError(
                "t2ramsey.make_cfg needs a pi2_pulse module (none produced or preset)"
            )
        readout = snapshot.module("opt_readout")
        if readout is None:
            raise RuntimeError(
                "t2ramsey.make_cfg needs a readout module (none produced or preset)"
            )
        raw_cfg = env.schema.lower_raw(ml, md=env.md)
        knobs = env.schema.lower(ml, md=env.md)
        t1 = _snapshot_float(snapshot, "t1", float(knobs["t1_seed_us"]))
        t2r = _snapshot_float(snapshot, "t2r", float(knobs["t2r_seed_us"]))
        relax_delay = _resolve_cfg_relax_delay(
            str(knobs["relax_delay_mode"]),
            t1=t1,
            fixed=float(knobs["relax_delay"]),
            knobs=knobs,
        )
        sweep_range = _resolve_cfg_sweep_range(
            str(knobs["sweep_range_mode"]),
            t2r=t2r,
            fixed=knobs["sweep_range"],
            knobs=knobs,
        )
        raw_cfg["modules"]["pi2_pulse"] = pi2_pulse
        raw_cfg["modules"]["readout"] = readout
        raw_cfg.pop("sweep", None)
        raw_cfg.pop("detune_ratio", None)
        raw_cfg["relax_delay"] = relax_delay
        raw_cfg["sweep_range"] = sweep_range
        return ml.make_cfg(raw_cfg, T2RamseyCfgTemplate)

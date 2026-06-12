"""t2echo — Hahn-echo Builder: acquire decay cosine → fit_decay_fringe → t2e.

Translates the notebook's T2EchoTask cfg_maker. Sets this flux point's value on
the picked flux device, sets up devices, acquires a decaying cosine fringe vs
delay time with ``ModularProgramV2`` (a Hahn-echo sequence), fits it with the
real ``fit_decay_fringe``, fills its sweep Result row in place, and returns the
raw t2e.

Unlike t2ramsey, the echo sequence refocuses static dephasing and typically
yields a longer coherence time; the difference is purely in the pulse sequence.
The same decay-cosine / fringe fit path applies regardless of the ``detune_ratio``
param (no branch).

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
from typing import Any, Optional, cast

import numpy as np

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.gui.app.autofluxdep.cfg import (
    FloatSpec,
    IntSpec,
    SweepSpec,
    SweepValue,
    flat_node_schema,
)
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema, sweepcfg_to_axis
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    SnrProbe,
    acquire_to_complex,
    axis_to_sweep,
    build_stop_checkers,
    is_good_fit,
    require_flux_device,
    set_flux_by_name,
    signal2real_flip,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
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
from zcu_tools.utils.fitting import fit_decay_fringe

logger = logging.getLogger(__name__)

_DEFAULT_T1 = 10.0  # us — smoothed t1 fallback
_DEFAULT_T2E = 5.0  # us — smoothed t2e fallback
_T2E_WINDOW_FACTOR = 2.5  # notebook: sweep_range = (0, 2.5 * prev_t2e)
_DEFAULT_DETUNE_RATIO = 0.2  # activate-detune fraction (one fringe per sweep)


def _default_t1() -> float:
    return _DEFAULT_T1


def _default_t2e() -> float:
    return _DEFAULT_T2E


def _placeholder_pi_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real (placeholder) module
    return {"type": "pi", "length": 0.1}


def _placeholder_pi2_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real (placeholder) module
    return {"type": "pi2", "length": 0.05}


def _default_readout() -> Any | None:
    return None


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
    """One flux point's t2echo: set flux → real acquire → fit_decay_fringe → Patch.

    Mirrors the lower-layer ``T2EchoTask`` ``measure_t2echo_fn`` + ``run``: a
    Hahn-echo sequence (pi/2 → τ/2 → pi → τ/2 → detuned pi/2) sweeps the total delay
    τ, the second pi/2 carries the activate-detune phase ramp, and
    ``fit_decay_fringe`` recovers T2Echo.
    """

    def __init__(self, env: RunEnv, builder: T2EchoBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot["t1"]  # optional smoothed t1 — relax_delay in make_cfg
        _ = snapshot["t2e"]  # smoothed (declared smooth="ewma") — sweep_range
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
        detune_ratio = self._builder.detune_ratio(env.schema)
        activate_detune = detune_ratio / length_sweep.step
        pi2_pulse = cfg.modules.pi2_pulse

        result.flux[idx] = env.flux

        probe = SnrProbe()

        def on_round(_round_count: int, avg_d: Any) -> None:
            signal = acquire_to_complex(avg_d)
            probe.value = signal
            np.copyto(result.signal[idx], signal2real_flip(signal))
            if env.round_hook is not None:
                env.round_hook(idx)

        stop_checkers = build_stop_checkers(env, probe, signal2real_flip)

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

        t2f, _, _, _, fit_curve, _ = fit_decay_fringe(times, real)

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
    """The t2echo provider — acquire decay cosine, real fit_decay_fringe, accumulating
    colormap.  Reports only the raw echo t2e (detune is refocused and not reported).
    """

    name = "t2echo"
    provides = ("t2e",)
    optional = (
        Dependency("t1", smooth="ewma", default=_default_t1),
        Dependency("t2e", smooth="ewma", default=_default_t2e),
    )
    requires_modules = (
        ModuleDep("pi_pulse", default=_placeholder_pi_pulse),
        ModuleDep("pi2_pulse", default=_placeholder_pi2_pulse),
    )
    optional_modules = (ModuleDep("opt_readout", default=_default_readout),)

    def make_default_schema(self) -> NodeCfgSchema:
        """The typed node-knob schema (defaults + types) — the param SSOT.

        Same shape as t2ramsey: ``sweep_range`` (``SweepSpec``) seeds the initial
        Result delay axis (the cfg's window is derived from the smoothed t2e in
        ``make_cfg``); ``detune_ratio`` defaults to the prototype's 0.2. The dead
        ``num_expts`` knob (never read) is dropped.
        """
        return NodeCfgSchema(
            flat_node_schema(
                (
                    (
                        "sweep_range",
                        SweepSpec(label="Delay time sweep (us)"),
                        SweepValue(start=0.0, stop=25.0, expts=121),
                    ),
                    (
                        "detune_ratio",
                        FloatSpec(label="Detune ratio"),
                        _DEFAULT_DETUNE_RATIO,
                    ),
                    ("reps", IntSpec(label="Reps"), 1000),
                    ("rounds", IntSpec(label="Rounds"), 10),
                    (
                        "earlystop_snr",
                        FloatSpec(label="Early-stop SNR", optional=True),
                        None,
                    ),
                )
            )
        )

    def detune_ratio(self, schema: NodeCfgSchema) -> float:
        """The activate-detune ratio for this placement (typed knob, default 0.2)."""
        return float(schema.lower(None)["detune_ratio"])

    def make_init_result(self, schema: NodeCfgSchema, flux: Any) -> Sweep1DResult:
        knobs = schema.lower(None)
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
        knobs = env.schema.lower(ml)
        cur_t1 = float(snapshot["t1"])  # smoothed t1
        prev_t2e = float(snapshot["t2e"])  # smoothed t2e
        return ml.make_cfg(
            {
                "modules": {
                    "pi_pulse": pi_pulse,
                    "pi2_pulse": pi2_pulse,
                    "readout": readout,
                },
                "relax_delay": max(1.0, 3.0 * cur_t1),
                "reps": knobs["reps"],
                "rounds": knobs["rounds"],
                "sweep_range": (0.0, _T2E_WINDOW_FACTOR * prev_t2e),
            },
            T2EchoCfgTemplate,
        )

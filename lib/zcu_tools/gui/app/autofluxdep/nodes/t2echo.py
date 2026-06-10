"""t2echo — Hahn-echo Builder: decay_cos → fit_decay_fringe → t2e.

Translates the notebook's T2EchoTask cfg_maker. Synthesises a decaying cosine
fringe vs delay time (t2 planted from the smoothed previous t2e * 1.1, fringe
frequency fixed at 0.3 1/us), fits it with the real ``fit_decay_fringe``, fills
its sweep Result row in place, and returns the raw t2e.

Unlike t2ramsey, the echo sequence refocuses static dephasing and typically
yields a longer coherence time, but the synthetic + fit path is identical (the
real-hardware difference is purely in the pulse sequence). The prototype
uniformly uses the decay-cosine / fringe path regardless of the ``detune_ratio``
param (no branch).

- needs the ``pi_pulse`` and ``pi2_pulse`` modules (lenrabi produces both) — the
  Hahn echo needs both a pi refocusing pulse and two pi/2 pulses. In the
  prototype both carry placeholder defaults, so they never actually skip;
  Phase B drops the defaults (real lenrabi output) to restore true skip.
- reads ``t1`` (smooth="ewma") and ``t2e`` (smooth="ewma") as optional deps:
  ``t2e`` seeds the planted t2 so the sweep tracks a plausible echo time;
  ``t1`` is available for cfg sanity checks (not used directly in the prototype).
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).

Phase B (cfg-builder): when the active context is configured — a populated
``ml`` + the upstream ``pi_pulse`` / ``pi2_pulse`` / ``opt_readout`` modules on
the snapshot (real ``PulseCfg`` / ``ReadoutCfg`` lenrabi/ro_optimize output) —
``produce`` lowers it into a runnable ``T2EchoCfgTemplate`` via
``ml.make_cfg`` (mirroring the notebook's T2EchoTask cfg_maker) and takes the
delay-time window (``sweep_range``) from the built cfg to parameterise the
synthetic acquire. The acquire is ALWAYS simulated (no hardware); routing
through ``make_cfg`` only exercises the real cfg pipeline and makes the cfg the
source of the measurement window. With the demo / empty-ml context (and the
existing run tests, which pass sentinel modules + no ml) the cfg is None and
produce keeps the pure snapshot/params simulation unchanged.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Mapping, Optional

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Decay1DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    accumulate_rounds,
    decay_cos,
    flux_drift,
    flux_snr,
    is_good_fit,
    parse_linear_axis,
    resolve_acquire_delay,
    resolve_rounds,
    signal_to_real,
)
from zcu_tools.program.v2 import ProgramV2Cfg
from zcu_tools.program.v2.modules import PulseCfg, ReadoutCfg, ResetCfg
from zcu_tools.utils.fitting import fit_decay_fringe

logger = logging.getLogger(__name__)

_DEFAULT_T1 = 10.0  # us — smoothed t1 fallback
_DEFAULT_T2E = 5.0  # us — smoothed t2e fallback
_FRINGE_FREQ = 0.3  # 1/us — fixed planted fringe frequency
_DEFAULT_SWEEP = (0.0, 25.0, 121)  # delay-time axis (us): start, stop, npts
_T2E_WINDOW_FACTOR = 2.5  # notebook: sweep_range = (0, 2.5 * prev_t2e)


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


def _default_readout() -> Optional[Any]:
    return None


def _is_lowerable_pulse(module: Any) -> bool:
    """Whether a resolved drive module is a concrete, lowerable ``PulseCfg``.

    A real lenrabi drive pulse is a ``PulseCfg`` (or its raw dict, ``type ==
    "pulse"``) and lowers into the run cfg. The prototype's placeholder
    ``{"type": "pi"/"pi2", "length": ...}`` is NOT a PulseCfg (it never
    validates), so this returns False there — the guard then keeps the pure
    synthetic path (no failed ``make_cfg``). Mirrors qubit_freq's guard naturally
    returning None in the prototype context.
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

    reset: Optional[ResetCfg] = None
    pi_pulse: PulseCfg
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class T2EchoCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base Hahn-echo cfg t2echo lowers a context into.

    ``ProgramV2Cfg`` (reps/rounds/relax) + the ``ExpCfgModel`` device/save fields
    + the t2echo modules and the ``sweep_range`` delay window — same bases as the
    lower-layer ``experiment/v2/autofluxdep`` ``T2EchoCfgTemplate``. The flux
    ``dev`` entry and the concrete ``length`` sweep are merged in by the
    lower-layer ``run`` (the GUI Builder only constructs this template); here
    ``produce`` reads the ``sweep_range`` window to parameterise the synthetic
    acquire.
    """

    modules: T2EchoModuleCfg
    sweep_range: tuple[float, float]


class T2EchoNode(Node):
    """One flux point's t2echo: synth decay-cos fringe → fit_decay_fringe → fill row → Patch."""

    def __init__(self, env: RunEnv, builder: "T2EchoBuilder") -> None:
        self._env = env
        self._builder = builder

    def _maybe_make_cfg(self, snapshot: Snapshot) -> Optional[T2EchoCfgTemplate]:
        """Build the run cfg when the context is configured for it, else None.

        ``make_cfg`` needs a populated ml + the upstream drive pulses
        (``pi_pulse`` / ``pi2_pulse``, real ``PulseCfg`` lenrabi output) + a
        readout (``opt_readout``); the default / demo context (empty ml, sentinel
        modules) has none, so produce keeps the pure snapshot-driven simulation
        there. No hardware is touched either way — Phase B simulates the acquire
        uniformly; routing through ``make_cfg`` (when configured) exercises the
        real cfg pipeline and makes the cfg the source of the delay-time window.
        """
        env = self._env
        if (
            env.ml is None
            or not _is_lowerable_pulse(snapshot.module("pi_pulse"))
            or not _is_lowerable_pulse(snapshot.module("pi2_pulse"))
            or snapshot.module("opt_readout") is None
        ):
            return None
        return self._builder.make_cfg(env, snapshot)

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot["t1"]  # optional smoothed t1 (available for cfg sanity)
        _ = float(
            snapshot["t2e"]
        )  # smoothed (declared smooth="ewma") — dependency contract
        _ = snapshot.module("pi_pulse")  # required module — prototype does not use
        _ = snapshot.module("pi2_pulse")  # required module — prototype does not use
        _ = snapshot.module("opt_readout")  # optional — prototype does not use

        result: Sweep1DResult = env.result

        # Build the run cfg from the active context (when configured) and take the
        # delay-time window from it; the acquire is SIMULATED below. With the demo
        # / empty-ml context the cfg is None and the delay axis is the param-driven
        # ``result.x`` (allocated at Run start) directly.
        cfg = self._maybe_make_cfg(snapshot)
        if cfg is not None:
            # the cfg's sweep_range = (0, 2.5 * smoothed_t2e) is the measurement
            # window; rebuild the delay axis over it (same point count as the
            # pre-allocated Result so the row shapes match)
            lo, hi = float(cfg.sweep_range[0]), float(cfg.sweep_range[1])
            times = np.linspace(lo, hi, result.n_x)
        else:
            times = result.x

        # t2e drifts parabolically with flux: ~6 us at sweet spot, up to ~21 us at
        # the edges. SNR varies sinusoidally to 0 at its troughs (dead points).
        # normalised sweep position (0→1): the drift/SNR shapes live on
        # [0,1], while real flux values are tiny — use the position so the
        # whole sweep spans one drift parabola and a few SNR cycles.
        pos = env.flux_idx / max(1, result.n_flux - 1)
        true_t2 = flux_drift(pos, baseline=6.0, amplitude=15.0)
        snr = flux_snr(pos)

        idx = env.flux_idx
        result.flux[idx] = env.flux

        base = env.flux_idx * 1000

        def make_round(k: int) -> NDArray[np.complex128]:
            return decay_cos(times, true_t2, _FRINGE_FREQ, snr=snr, seed=base + k)

        def on_round(avg: NDArray[np.complex128], _k: int) -> None:
            np.copyto(result.signal[idx], signal_to_real(avg))
            if env.round_hook is not None:
                env.round_hook(idx)

        averaged = accumulate_rounds(
            make_round,
            resolve_rounds(env.params),
            on_round,
            delay=resolve_acquire_delay(env.params),
        )
        real = signal_to_real(averaged)

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

        logger.debug(
            "t2echo fit @flux%d: t2e=%.3f us (plant t2=%.3f)",
            idx,
            float(t2f),
            true_t2,
        )

        patch = Patch()
        patch.set("t2e", float(t2f))
        return patch


class T2EchoBuilder(Builder):
    """The t2echo provider — decay-cosine synth, real fit_decay_fringe, accumulating
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
    base_params = (
        "sweep_range",
        "num_expts",
        "detune_ratio",
        "reps",
        "rounds",
        "earlystop_snr",
        "acquire_delay",
    )

    def make_init_result(self, params: Mapping[str, Any], flux: Any) -> Sweep1DResult:
        times = parse_linear_axis(params.get("sweep_range"), _DEFAULT_SWEEP)
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
        needs concrete drive pulses (Fast Fail), unlike the synthetic path which
        fabricates a signal.
        """
        params = env.params
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
                "reps": int(params.get("reps", 1000)),
                "rounds": int(params.get("rounds", 10)),
                "sweep_range": (0.0, _T2E_WINDOW_FACTOR * prev_t2e),
            },
            T2EchoCfgTemplate,
        )

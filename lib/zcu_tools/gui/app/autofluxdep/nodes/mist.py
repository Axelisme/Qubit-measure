"""mist — 1D gain-sweep Builder: variance curve readout (no fit).

Synthesises a state-disturbance curve (``variance_curve``) over a gain axis,
reads the variance directly — there is no fit step — and fills its Sweep1DResult
row in place. ``fit_value`` and ``fit_curve`` remain nan (allocated as nan by
``Sweep1DResult.allocate``); the ``ColormapLinePlotter`` shows the flux × gain
colormap with the latest flux rows as traces (no fit marker).

- needs the ``pi_pulse`` module (pi-pulse prepares the excited state whose
  disturbance the variance measures). In the prototype it carries a placeholder
  default, so it never actually skips; Phase B drops the default to restore true
  skip-if-absent.
- the ``opt_readout`` module is optional (ro_optimize produces it); used by the
  cfg builder as the run cfg's ``readout`` when the context is configured.

No fit step: the variance curve is a monotone logistic ramp; MIST reads the
variance magnitude directly (the real pipeline reads IQ scatter). The row is
considered complete after one compute step, so ``round_hook`` is called once.
Provides ``success=1.0`` (float, consistent with the info-value domain) to signal
that the MIST pass completed without a hardware error.

Phase B cfg-builder (mirrors qubit_freq): when the active context is configured
(a populated ml + a ``pi_pulse`` module on the snapshot + the mist-drive "設定頭"
params present) ``produce`` lowers it into the lower-layer ``MistCfgTemplate``
(``ProgramV2Cfg`` + ``ExpCfgModel`` with ``pi_pulse`` / ``mist_pulse`` / ``readout``
modules) via ``ml.make_cfg`` — exercising the real cfg pipeline — and takes the
mist-drive onset gain from the built cfg's ``mist_pulse.gain``. The acquire is
ALWAYS simulated (no ``ModularProgramV2`` / no ``soc.acquire``); with the demo /
empty-ml context the cfg is None and the onset falls back to the fixed prototype
value. There is no fit either way — mist reads the variance directly.
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
from zcu_tools.gui.app.autofluxdep.nodes.plotters import ColormapLinePlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    accumulate_rounds,
    parse_linear_axis,
    resolve_acquire_delay,
    resolve_rounds,
    variance_curve,
)
from zcu_tools.program.v2 import ProgramV2Cfg

logger = logging.getLogger(__name__)

_DEFAULT_GAIN_SWEEP: tuple[float, float, int] = (0.0, 1.0, 51)
_ONSET_GAIN = 0.5  # fixed onset for the prototype variance curve (empty-ml fallback)


class MistModuleCfg(ConfigBase):
    """The mist run cfg's module set, mirroring the lower-layer ``MistModuleCfg``.

    ``pi_pulse`` prepares the excited state, ``mist_pulse`` is the disturbance
    drive whose gain the experiment sweeps, and ``readout`` reads the resulting
    state. ``reset`` is optional (none in the prototype). Typed loosely as ``Any``
    here — ``ml.make_cfg`` lowers the raw dicts/modules into the concrete
    PulseCfg / ReadoutCfg via the module factory; the lower-layer
    ``experiment/v2/autofluxdep`` ``MistCfgTemplate`` carries the strict types.
    """

    reset: Optional[Any] = None
    pi_pulse: Any
    mist_pulse: Any
    readout: Any


class MistCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base cfg mist lowers a context into (no sweep / dev yet).

    Mirrors the lower-layer ``experiment/v2/autofluxdep`` ``MistCfgTemplate``
    (``ProgramV2Cfg`` reps/rounds/relax + ``ExpCfgModel`` device fields) with the
    ``pi_pulse`` / ``mist_pulse`` / ``readout`` modules. The flux ``dev`` entry and
    the ``gain`` sweep (which sweeps ``mist_pulse.gain``) are merged in by the run
    layer downstream, not built here — exactly as the lower layer's ``run`` does.
    """

    modules: MistModuleCfg


# --- placeholder external bindings (Phase B: inject from project/metadata) ---
def _placeholder_pi_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real module
    return {"type": "pi", "length": 0.1}


def _default_opt_readout() -> Optional[Any]:
    # last-resort readout if neither a Node produced one nor ml has a preset.
    return None


class MistNode(Node):
    """One flux point's MIST: synth variance curve → fill row → Patch.

    No fit step: ``variance_curve`` returns a real (n_gain,) magnitude directly.
    ``fit_value[idx]`` and ``fit_curve[idx]`` are left as nan (already the
    allocate default), so the ``ColormapLinePlotter`` shows only the colormap
    (no fit marker). Calls ``round_hook`` once per round while filling the row.

    When the active context is configured, ``produce`` first lowers it into the
    run cfg via the Builder's ``make_cfg`` (exercising the real ml pipeline) and
    takes the disturbance onset gain from the built cfg's ``mist_pulse.gain``;
    otherwise (demo / empty-ml) it falls back to the fixed prototype onset. The
    acquire is SIMULATED either way — no hardware is touched.
    """

    def __init__(self, env: RunEnv, builder: "MistBuilder") -> None:
        self._env = env
        self._builder = builder

    def _maybe_make_cfg(self, snapshot: Snapshot) -> Optional[MistCfgTemplate]:
        """Build the run cfg when the context is configured for it, else None.

        ``make_cfg`` needs a ``pi_pulse`` module + the mist-drive params; the
        default / demo context (empty ml) has neither, so produce keeps the pure
        snapshot-driven simulation there. No hardware is touched either way —
        Phase B simulates the acquire uniformly; routing through ``make_cfg``
        (when configured) exercises the real cfg pipeline and makes the cfg the
        source of the disturbance onset gain.
        """
        env = self._env
        if (
            env.ml is None
            or snapshot.module("pi_pulse") is None
            or not env.params.get("mist_waveform")
            or env.params.get("mist_ch") is None
        ):
            return None
        return self._builder.make_cfg(env, snapshot)

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

        _ = snapshot.module("pi_pulse")  # required — consumed by real hardware
        _ = snapshot.module("opt_readout")  # optional — optimised readout preset

        # Build the run cfg from the active context (when configured) and take the
        # disturbance onset gain from it; the acquire is SIMULATED below. With the
        # demo / empty-ml context the cfg is None and the onset is the fixed
        # prototype value (the synthetic variance curve's logistic step centre).
        cfg = self._maybe_make_cfg(snapshot)
        onset = float(cfg.modules.mist_pulse.gain) if cfg is not None else _ONSET_GAIN

        result: Sweep1DResult = env.result
        gains = result.x

        idx = env.flux_idx
        result.flux[idx] = env.flux
        # fit_value[idx] and fit_curve[idx] remain nan — mist has no fit scalar

        base = env.flux_idx * 1000

        def make_round(k: int) -> NDArray[np.float64]:
            return variance_curve(gains, onset, seed=base + k)

        def on_round(avg: NDArray[np.float64], _k: int) -> None:
            np.copyto(result.signal[idx], avg)
            if env.round_hook is not None:
                env.round_hook(idx)

        curve = accumulate_rounds(
            make_round,
            resolve_rounds(env.params),
            on_round,
            delay=resolve_acquire_delay(env.params),
        )

        logger.debug(
            "mist @flux%d: success, variance range [%.3f, %.3f] (onset=%.3f)",
            idx,
            float(curve.min()),
            float(curve.max()),
            onset,
        )

        patch = Patch()
        patch.set("success", 1.0)  # float: consistent with info-value domain
        return patch


class MistBuilder(Builder):
    """The MIST provider — variance-curve synth, no fit, accumulating colormap.

    Sweeps a gain axis per flux point, synthesises a state-disturbance curve, and
    records the variance directly (no fit). ``fit_value`` stays nan so the
    ``ColormapLinePlotter`` renders only the flux×gain colormap. Provides ``success``
    (float 1.0) to signal that the MIST pass completed; the ``opt_readout``
    module is consumed (from ro_optimize) to configure the readout during the real
    measurement.
    """

    name = "mist"
    provides = ("success",)
    provides_modules: tuple[str, ...] = ()
    requires_modules = (ModuleDep("pi_pulse", default=_placeholder_pi_pulse),)
    optional_modules = (ModuleDep("opt_readout", default=_default_opt_readout),)
    base_params = (
        "gain_sweep",
        "reps",
        "rounds",
        "relax_delay",
        "acquire_delay",
        # the mist disturbance pulse "設定頭" — what the cfg builder lowers into
        # mist_pulse (the experiment then sweeps mist_pulse.gain; this base gain is
        # the onset the synthetic variance curve uses when the cfg drives produce)
        "mist_waveform",
        "mist_ch",
        "mist_nqz",
        "mist_freq",
        "mist_gain",
        "mist_length",
    )

    def make_init_result(self, params: Mapping[str, Any], flux: Any) -> Sweep1DResult:
        gains = parse_linear_axis(params.get("gain_sweep"), _DEFAULT_GAIN_SWEEP)
        return Sweep1DResult.allocate(flux, gains, x_label="gain")

    def make_plotter(self, figure: Any) -> ColormapLinePlotter:
        return ColormapLinePlotter(
            figure, title="mist", y_label="Readout Gain (a.u.)", num_lines=1
        )

    def build_node(self, env: RunEnv) -> MistNode:
        return MistNode(env, self)

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> MistCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the lower-layer ``experiment/v2/autofluxdep`` mist ``cfg_maker``
        (the notebook keeps mist commented out; the lower-layer ``MistTask`` is the
        ground truth): the ``pi_pulse`` is the latest-available pi-pulse module, the
        ``readout`` is the latest-available optimised readout, and the disturbance
        ``mist_pulse`` waveform / channel / gain / nqz / freq come from the node's
        params (the "設定頭"). The flux ``dev`` entry and the ``gain`` sweep (which
        sweeps ``mist_pulse.gain``) are NOT here — the run layer merges them, exactly
        as the lower layer's ``run`` does.

        Raises if the ``pi_pulse`` module is unavailable or the mist-drive params
        are unset — a real run needs a concrete disturbance pulse + an excited-state
        preparation pulse (Fast Fail), unlike the synthetic path which fabricates a
        variance curve.
        """
        params = env.params
        ml = env.ml
        if ml is None:
            raise RuntimeError("mist.make_cfg needs an active ModuleLibrary")
        pi_pulse = snapshot.module("pi_pulse")
        if pi_pulse is None:
            raise RuntimeError(
                "mist.make_cfg needs a pi_pulse module (none produced or preset)"
            )
        # readout is optional in the snapshot (ro_optimize → ml preset → default);
        # when absent the lower-layer cfg's required `readout` field cannot be filled,
        # so a real run needs one — but _maybe_make_cfg only routes here once a
        # readout-bearing context exists. Fall back to the ml preset path is the
        # caller's responsibility; here we use whatever the snapshot resolved.
        readout = snapshot.module("opt_readout")
        if readout is None:
            raise RuntimeError(
                "mist.make_cfg needs a readout module (none produced or preset)"
            )
        waveform_name = params.get("mist_waveform")
        ch = params.get("mist_ch")
        if not waveform_name or ch is None:
            raise RuntimeError("mist.make_cfg needs mist_waveform + mist_ch params set")
        return ml.make_cfg(
            {
                "modules": {
                    "pi_pulse": pi_pulse,
                    "mist_pulse": {
                        "type": "pulse",
                        "waveform": ml.get_waveform(
                            waveform_name,
                            {"length": float(params.get("mist_length", 0.1))},
                        ),
                        "ch": int(ch),
                        "nqz": int(params.get("mist_nqz", 2)),
                        "gain": float(params.get("mist_gain", 0.5)),
                        "freq": float(params.get("mist_freq", 0.0)),
                    },
                    "readout": readout,
                },
                "relax_delay": float(params.get("relax_delay", 0.5)),
                "reps": int(params.get("reps", 1000)),
                "rounds": int(params.get("rounds", 100)),
            },
            MistCfgTemplate,
        )

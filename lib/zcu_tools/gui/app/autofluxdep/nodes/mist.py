"""mist — 1D gain-sweep Builder: variance curve readout (no fit).

Sets this flux point's value on the picked flux device, sets up devices, acquires
a state-disturbance curve over a gain axis with ``ModularProgramV2`` (Reset →
pi_pulse → mist_pulse → Readout), reads the variance directly — there is no fit
step — and fills its Sweep1DResult row in place. ``fit_value`` and ``fit_curve``
remain nan (allocated as nan by ``Sweep1DResult.allocate``); the
``ColormapLinePlotter`` shows the flux × gain colormap with the latest flux rows
as traces (no fit marker).

- needs the ``pi_pulse`` module (pi-pulse prepares the excited state whose
  disturbance the variance measures). It carries a placeholder default, so it
  never actually skips when lenrabi is absent.
- the ``opt_readout`` module is optional (ro_optimize produces it); used by the
  cfg builder as the run cfg's ``readout``.

No fit step: MIST reads the disturbance magnitude directly from the acquired IQ
scatter. The row is considered complete after one round, so ``round_hook`` is
called once per round. Provides ``success=1.0`` (float, consistent with the
info-value domain) to signal that the MIST pass completed without a hardware
error.

``produce`` lowers the active context (a populated ml + a ``pi_pulse`` module on
the snapshot + the mist-drive "設定頭" params present) into the lower-layer
``MistCfgTemplate`` (``ProgramV2Cfg`` + ``ExpCfgModel`` with ``pi_pulse`` /
``mist_pulse`` / ``readout`` modules) via ``ml.make_cfg``, then acquires against a
flux-aware MockSoc (offline) or real hardware. ``make_cfg`` Fast Fails when the
context is unconfigured. There is no fit — mist reads the variance directly.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    acquire_to_complex,
    axis_to_sweep,
    parse_linear_axis,
    require_flux_device,
    set_flux_by_name,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.plotters import ColormapLinePlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import ModuleDep
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    Readout,
    Reset,
    sweep2param,
)

logger = logging.getLogger(__name__)

_DEFAULT_GAIN_SWEEP: tuple[float, float, int] = (0.0, 1.0, 51)


def _mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """The MIST state-disturbance magnitude (lower-layer ``mist_signal2real``).

    The disturbance is read directly from the IQ scatter — no fit: the centred
    magnitude normalised by its own spread, so a flat (undisturbed) trace reads
    near zero and the onset shows as the magnitude rising past a gain threshold."""
    if np.all(np.isnan(signals)):
        return np.abs(signals)
    mag = np.abs(signals - np.mean(signals))
    std = float(np.std(mag))
    return mag / (std + 1e-12)


class MistModuleCfg(ConfigBase):
    """The mist run cfg's module set, mirroring the lower-layer ``MistModuleCfg``.

    ``pi_pulse`` prepares the excited state, ``mist_pulse`` is the disturbance
    drive whose gain the experiment sweeps, and ``readout`` reads the resulting
    state. ``reset`` is optional (none in the prototype). Typed loosely as ``Any``
    here — ``ml.make_cfg`` lowers the raw dicts/modules into the concrete
    PulseCfg / ReadoutCfg via the module factory; the lower-layer
    ``experiment/v2/autofluxdep`` ``MistCfgTemplate`` carries the strict types.
    """

    reset: Any | None = None
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


def _default_opt_readout() -> Any | None:
    # last-resort readout if neither a Node produced one nor ml has a preset.
    return None


class MistNode(Node):
    """One flux point's MIST: set flux → real acquire → variance curve → fill row → Patch.

    No fit step: ``_mist_signal2real`` returns a real (n_gain,) magnitude directly.
    ``fit_value[idx]`` and ``fit_curve[idx]`` are left as nan (already the
    allocate default), so the ``ColormapLinePlotter`` shows only the colormap
    (no fit marker). Calls ``round_hook`` once per round while filling the row.

    ``produce`` lowers the active context into the run cfg via the Builder's
    ``make_cfg`` (Fast Fail if unconfigured), sweeps the disturbance pulse gain,
    and acquires against a flux-aware MockSoc (offline) or real hardware.
    """

    def __init__(self, env: RunEnv, builder: MistBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

        _ = snapshot.module("pi_pulse")  # required — excited-state preparation
        _ = snapshot.module("opt_readout")  # required — readout

        result: Sweep1DResult = env.result
        gains = result.x
        idx = env.flux_idx

        # Lower the active context into the run cfg (Fast Fail if unconfigured: a
        # real acquire needs concrete pi_pulse + mist_pulse + readout). MIST is a
        # known SimEngine gap — the .acquire below raises (not swallowed) under the
        # mock if the engine cannot model the disturbance program, surfacing the
        # gap rather than degrading to noise.
        cfg = self._builder.make_cfg(env, snapshot)

        flux_device = require_flux_device(env, "mist")
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

        # Sweep the disturbance pulse gain over the Result's gain axis (lower layer
        # sets sweep2param("gain") on mist_pulse).
        gain_sweep = axis_to_sweep(gains)
        cfg.modules.mist_pulse.set_param("gain", sweep2param("gain", gain_sweep))

        result.flux[idx] = env.flux
        # fit_value[idx] and fit_curve[idx] remain nan — mist has no fit scalar

        def on_round(_round_count: int, avg_d: Any) -> None:
            np.copyto(result.signal[idx], _mist_signal2real(acquire_to_complex(avg_d)))
            if env.round_hook is not None:
                env.round_hook(idx)

        stop_checkers: list[Any] = []
        if env.should_stop is not None:
            stop_checkers.append(env.should_stop)

        raw = ModularProgramV2(
            env.soccfg,
            cfg,
            modules=[
                Reset("reset", cfg.modules.reset),
                Pulse("pi_pulse", cfg.modules.pi_pulse),
                Pulse("mist_pulse", cfg.modules.mist_pulse),
                Readout("readout", cfg.modules.readout),
            ],
            sweep=[("gain", gain_sweep)],
        ).acquire(
            env.soc,
            progress=False,
            round_hook=on_round,
            stop_checkers=stop_checkers,
        )
        curve = _mist_signal2real(acquire_to_complex(raw))
        np.copyto(result.signal[idx], curve)

        logger.debug(
            "mist @flux%d: success, variance range [%.3f, %.3f]",
            idx,
            float(curve.min()),
            float(curve.max()),
        )

        patch = Patch()
        patch.set("success", 1.0)  # float: consistent with info-value domain
        return patch


class MistBuilder(Builder):
    """The MIST provider — acquire variance curve, no fit, accumulating colormap.

    Sweeps a gain axis per flux point, acquires a state-disturbance curve, and
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
        # the mist disturbance pulse "設定頭" — what the cfg builder lowers into
        # mist_pulse (the experiment then sweeps mist_pulse.gain over the gain axis)
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
        preparation pulse (Fast Fail).
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

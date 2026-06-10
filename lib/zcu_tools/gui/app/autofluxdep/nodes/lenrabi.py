"""lenrabi — length-Rabi Builder: rabi_oscillation → fit_rabi → pi/pi2 lengths.

Translates the notebook's LenRabiTask cfg_maker. Synthesises a cosine Rabi
oscillation vs pulse length (rabi_freq planted near 0.5 1/us), fits it with the
real ``fit_rabi``, fills its sweep Result row in place, and returns the raw pi
and pi2 lengths plus the Rabi frequency.

- requires ``qubit_freq`` (a hard require via Dependency): the Rabi experiment
  drives the qubit on resonance, so no qubit frequency → no sensible cfg.
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).
- provides the ``pi_pulse`` and ``pi2_pulse`` modules (placeholder dicts in the
  prototype; the real impl would fill proper PulseReadoutCfg objects).

Phase B (cfg-builder): when the context is configured (a populated ml + an
``opt_readout`` module + the drive "設定頭" params), ``produce`` lowers it into a
real ``LenRabiCfgTemplate`` via ``Builder.make_cfg`` → ``ml.make_cfg`` — mirroring
the notebook's ``cfg_maker`` and the lower-layer ``experiment/v2/autofluxdep``
LenRabiCfgTemplate — exercising the real cfg pipeline. The acquire stays SIMULATED
either way (no hardware): with the demo / empty-ml context the cfg is None and
produce keeps the pure snapshot-driven simulation unchanged. Compare
``notebook_md/autofluxdep.md`` (the LenRabiTask block).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.plotters import ColormapLinePlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    accumulate_rounds,
    flux_drift,
    flux_snr,
    is_good_fit,
    parse_linear_axis,
    rabi_oscillation,
    resolve_acquire_delay,
    resolve_rounds,
    signal_to_real,
)
from zcu_tools.program.v2 import ProgramV2Cfg, PulseCfg, ReadoutCfg, ResetCfg
from zcu_tools.utils.fitting import fit_rabi

logger = logging.getLogger(__name__)


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
    concrete ``length`` sweep are merged in by the lower-layer's ``run`` (and, in
    the GUI prototype, the sweep is simulated from ``sweep_range`` directly); they
    are NOT part of the template, exactly like qubit_freq's detune sweep.
    """

    modules: LenRabiModuleCfg
    sweep_range: tuple[float, float]


def _last_fit(result: Any) -> float:
    """Return the last non-nan fit_value (the most recent pi_length)."""
    valid = result.fit_value[~np.isnan(result.fit_value)]
    return float(valid[-1]) if valid.size else float("nan")


_DEFAULT_SWEEP = (0.0, 6.0, 121)  # pulse-length axis (us): start, stop, npts


def _default_readout() -> Any | None:
    return None


class LenRabiNode(Node):
    """One flux point's lenrabi: synth Rabi oscillation → fit_rabi → fill row → Patch."""

    def __init__(self, env: RunEnv, builder: LenRabiBuilder) -> None:
        self._env = env
        self._builder = builder

    def _maybe_make_cfg(self, snapshot: Snapshot) -> LenRabiCfgTemplate | None:
        """Build the run cfg when the context is configured for it, else None.

        ``make_cfg`` needs a readout module (``opt_readout``) + the drive params;
        the default / demo context (empty ml) has neither, so produce keeps the
        pure snapshot-driven simulation there. No hardware is touched either way —
        Phase B simulates the acquire uniformly; routing through ``make_cfg`` (when
        configured) exercises the real cfg pipeline and makes the cfg the source of
        the drive on-resonance frequency. Mirrors qubit_freq's guard.
        """
        env = self._env
        if (
            env.ml is None
            or snapshot.module("opt_readout") is None
            or not env.params.get("qub_waveform")
            or env.params.get("qub_ch") is None
        ):
            return None
        return self._builder.make_cfg(env, snapshot)

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot["qubit_freq"]  # required — the on-resonance drive frequency
        _ = snapshot.module("opt_readout")  # optional — readout for the cfg path

        # Build the run cfg from the active context (when configured); the acquire
        # is SIMULATED below. The cfg's rabi_pulse is the on-resonance drive — its
        # freq is the (required) qubit frequency — so going through make_cfg
        # exercises the real cfg pipeline without changing the simulated physics
        # (the Rabi oscillation is a function of pulse LENGTH, not the drive freq,
        # so the centre value the cfg carries is the length extent, applied to the
        # Result axis already). With the demo / empty-ml context the cfg is None
        # and produce stays purely synthetic. Mirrors qubit_freq's wiring.
        _ = self._maybe_make_cfg(snapshot)

        result: Sweep1DResult = env.result
        lengths = result.x

        # rabi_freq drifts parabolically with flux: ~0.5 1/us at sweet spot, up to
        # ~0.9 1/us at the edges. SNR varies sinusoidally to 0 at its troughs.
        # normalised sweep position (0→1): the drift/SNR shapes live on
        # [0,1], while real flux values are tiny — use the position so the
        # whole sweep spans one drift parabola and a few SNR cycles.
        pos = env.flux_idx / max(1, result.n_flux - 1)
        rabi_freq = flux_drift(pos, baseline=0.5, amplitude=0.4)
        snr = flux_snr(pos)

        idx = env.flux_idx
        result.flux[idx] = env.flux

        base = env.flux_idx * 1000

        def make_round(k: int) -> NDArray[np.complex128]:
            return rabi_oscillation(lengths, rabi_freq, snr=snr, seed=base + k)

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

        pi_x, _, pi2_x, _, freq, _, fit_curve, _ = fit_rabi(lengths, real)

        if not is_good_fit(real, fit_curve):
            logger.debug("lenrabi fit @flux%d: poor fit (SNR-trough?) — discarded", idx)
            if env.round_hook is not None:
                env.round_hook(idx)  # raw row already shown; fit fields stay nan
            return Patch()  # partial: omit pi_length/pi2_length/rabi_freq + modules → downstream fallback

        result.fit_value[idx] = float(pi_x)
        np.copyto(result.fit_curve[idx], np.asarray(fit_curve, dtype=np.float64))
        if env.round_hook is not None:
            env.round_hook(idx)

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
        patch.set_module("pi_pulse", {"type": "pi", "length": float(pi_x)})
        patch.set_module("pi2_pulse", {"type": "pi2", "length": float(pi2_x)})
        return patch


class LenRabiBuilder(Builder):
    """The lenrabi provider — Rabi-oscillation synth, real fit_rabi, accumulating
    colormap.  Produces pi_pulse and pi2_pulse module placeholders in addition to
    the scalar pi_length / pi2_length / rabi_freq info values.
    """

    name = "lenrabi"
    provides = ("pi_length", "pi2_length", "rabi_freq")
    provides_modules = ("pi_pulse", "pi2_pulse")
    requires = (Dependency("qubit_freq"),)
    optional_modules = (ModuleDep("opt_readout", default=_default_readout),)
    base_params = (
        "sweep_range",
        "num_expts",
        "reps",
        "rounds",
        "relax_delay",
        "earlystop_snr",
        "acquire_delay",
        # the drive pulse "設定頭" — what the cfg builder lowers into rabi_pulse
        # (freq comes from the required qubit_freq, readout from the snapshot)
        "qub_waveform",
        "qub_ch",
        "qub_nqz",
        "qub_gain",
        "qub_length",
    )

    def make_init_result(self, params: Mapping[str, Any], flux: Any) -> Sweep1DResult:
        lengths = parse_linear_axis(params.get("sweep_range"), _DEFAULT_SWEEP)
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
        The pulse-length ``sweep_range`` is taken from the already-allocated Result
        trailing axis so the cfg's swept extent matches the simulated length axis
        (the notebook computes ``(0.05, max(5*prev_pi_len, 0.5))``; the GUI
        prototype's extent is the user-tuned ``sweep_range`` param). The flux ``dev``
        entry and the concrete ``length`` sweep are NOT here — the lower-layer's
        ``run`` merges them, exactly like qubit_freq's detune.

        Raises if the readout module is unavailable or the drive params are unset —
        a real run needs a concrete drive pulse (Fast Fail), unlike the synthetic
        path which fabricates a signal.
        """
        params = env.params
        ml = env.ml
        if ml is None:
            raise RuntimeError("lenrabi.make_cfg needs an active ModuleLibrary")
        readout = snapshot.module("opt_readout")
        if readout is None:
            raise RuntimeError(
                "lenrabi.make_cfg needs a readout module (none produced or preset)"
            )
        waveform_name = params.get("qub_waveform")
        ch = params.get("qub_ch")
        if not waveform_name or ch is None:
            raise RuntimeError(
                "lenrabi.make_cfg needs qub_waveform + qub_ch params set"
            )
        qubit_freq = float(snapshot["qubit_freq"])

        # the pulse-length extent (start, stop): the simulated trailing axis when a
        # Result is allocated, else the parsed sweep_range param (so make_cfg works
        # standalone, e.g. in tests, without a Result curried in).
        if env.result is not None:
            xs = np.asarray(env.result.x, dtype=np.float64)
        else:
            xs = parse_linear_axis(params.get("sweep_range"), _DEFAULT_SWEEP)
        sweep_range = (float(xs[0]), float(xs[-1]))

        return ml.make_cfg(
            {
                "modules": {
                    "rabi_pulse": {
                        "type": "pulse",
                        "waveform": ml.get_waveform(
                            waveform_name,
                            {"length": float(params.get("qub_length", 0.1))},
                        ),
                        "ch": int(ch),
                        "nqz": int(params.get("qub_nqz", 2)),
                        "gain": float(params.get("qub_gain", 0.05)),
                        "freq": qubit_freq,
                    },
                    "readout": readout,
                },
                "relax_delay": float(params.get("relax_delay", 0.5)),
                "reps": int(params.get("reps", 1000)),
                "rounds": int(params.get("rounds", 10)),
                "sweep_range": sweep_range,
            },
            LenRabiCfgTemplate,
        )

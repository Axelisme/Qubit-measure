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
from zcu_tools.gui.app.autofluxdep.cfg import (
    FloatSpec,
    IntSpec,
    SweepSpec,
    SweepValue,
    node_field,
    node_section,
    sectioned_node_schema,
    str_scalar_spec,
)
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
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import READOUT_LIBRARY_ALIASES
from zcu_tools.gui.app.autofluxdep.nodes.plotters import ColormapLinePlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.waveform_defaults import waveform_or_const
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

_DEFAULT_QUB_WAVEFORM = "qub_flat"
_DEFAULT_QUB_CH = 0
_DEFAULT_RELAX_DELAY = 30.0
_DEFAULT_EARLYSTOP_SNR = 30.0


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


def _drive_pulse_with_length(base: PulseCfg, length: float) -> dict[str, Any]:
    pulse = base.model_copy(deep=True)
    pulse.set_param("length", float(length))
    return pulse.to_dict()


class LenRabiNode(Node):
    """One flux point's lenrabi: set flux → real acquire → fit_rabi → fill row → Patch.

    Mirrors the lower-layer ``LenRabiTask`` ``measure_fn`` + ``run``: the on-resonance
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
        lengths = result.x
        idx = env.flux_idx

        # Lower the active context into the run cfg (Fast Fail if unconfigured: a
        # real acquire needs a concrete readout + on-resonance drive pulse).
        cfg = self._builder.make_cfg(env, snapshot)

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
        patch.set_module("pi_pulse", _drive_pulse_with_length(base_drive_pulse, pi_x))
        patch.set_module("pi2_pulse", _drive_pulse_with_length(base_drive_pulse, pi2_x))
        return patch


class LenRabiBuilder(Builder):
    """The lenrabi provider — acquire Rabi oscillation, real fit_rabi, accumulating
    colormap.  Produces pi_pulse and pi2_pulse modules in addition to the scalar
    pi_length / pi2_length / rabi_freq info values.
    """

    name = "lenrabi"
    provides = ("pi_length", "pi2_length", "rabi_freq")
    provides_modules = ("pi_pulse", "pi2_pulse")
    requires = (Dependency("qubit_freq"),)
    optional_modules = (
        ModuleDep(
            "opt_readout", default=_default_readout, aliases=READOUT_LIBRARY_ALIASES
        ),
    )

    def make_default_schema(self) -> NodeCfgSchema:
        """The typed node-knob schema (defaults + types) — the param SSOT.

        ``sweep_range`` is the pulse-length axis as a ``SweepSpec`` (expts-defined,
        like qubit_freq's detune). Its default starts away from zero and spans the
        notebook's conservative initial ``5 * pi_len`` window for a 0.1 us pi
        guess. Drive defaults follow the measure-gui qubit-pulse convention:
        ``qub_flat`` on channel 0; missing named waveforms lower to an inline const
        pulse with ``qub_length``. The prototype's dead ``num_expts`` knob (never
        read — the point count came from the axis itself) is dropped.
        """
        return sectioned_node_schema(
            (
                node_section(
                    "sweep",
                    "Sweep",
                    node_field(
                        "sweep_range",
                        "length",
                        SweepSpec(label="Pulse length sweep (us)"),
                        SweepValue(start=0.05, stop=0.5, expts=101),
                    ),
                ),
                node_section(
                    "acquire",
                    "Acquisition",
                    node_field(
                        "reps",
                        "reps",
                        IntSpec(label="Reps"),
                        1000,
                    ),
                    node_field(
                        "rounds",
                        "rounds",
                        IntSpec(label="Rounds"),
                        10,
                    ),
                    node_field(
                        "relax_delay",
                        "relax_delay",
                        FloatSpec(label="Relax delay (us)"),
                        _DEFAULT_RELAX_DELAY,
                    ),
                    node_field(
                        "earlystop_snr",
                        "earlystop_snr",
                        FloatSpec(label="Early-stop SNR", optional=True),
                        _DEFAULT_EARLYSTOP_SNR,
                    ),
                ),
                node_section(
                    "drive",
                    "Drive pulse",
                    node_field(
                        "qub_waveform",
                        "waveform",
                        str_scalar_spec("Drive waveform", optional=True),
                        _DEFAULT_QUB_WAVEFORM,
                    ),
                    node_field(
                        "qub_ch",
                        "ch",
                        IntSpec(label="Drive ch", optional=True),
                        _DEFAULT_QUB_CH,
                    ),
                    node_field(
                        "qub_nqz",
                        "nqz",
                        IntSpec(label="Drive nqz"),
                        2,
                    ),
                    node_field(
                        "qub_gain",
                        "gain",
                        FloatSpec(label="Drive gain"),
                        0.05,
                    ),
                    node_field(
                        "qub_length",
                        "length",
                        FloatSpec(label="Drive length (us)"),
                        0.1,
                    ),
                ),
            )
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
        The pulse-length ``sweep_range`` is taken from the already-allocated Result
        trailing axis so the cfg's swept extent matches the acquired length axis
        (the notebook computes ``(0.05, max(5*prev_pi_len, 0.5))``; the GUI's
        extent is the user-tuned ``sweep_range`` param). The flux ``dev`` entry and
        the concrete ``length`` sweep are NOT here — ``produce`` merges them,
        exactly like qubit_freq's detune.

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
        knobs = env.schema.lower(ml, md=env.md)
        waveform_name = knobs.get("qub_waveform")
        ch = knobs.get("qub_ch")
        if ch is None:
            raise RuntimeError("lenrabi.make_cfg needs qub_ch param set")
        qubit_freq = float(snapshot["qubit_freq"])

        # the pulse-length extent (start, stop): the Result's trailing axis when a
        # Result is allocated, else the lowered sweep_range axis (so make_cfg works
        # standalone, e.g. in tests, without a Result curried in).
        if env.result is not None:
            xs = np.asarray(env.result.x, dtype=np.float64)
        else:
            xs = sweepcfg_to_axis(knobs["sweep_range"])
        sweep_range = (float(xs[0]), float(xs[-1]))

        return ml.make_cfg(
            {
                "modules": {
                    "rabi_pulse": {
                        "type": "pulse",
                        "waveform": waveform_or_const(
                            ml, waveform_name, length=knobs["qub_length"]
                        ),
                        "ch": ch,
                        "nqz": knobs["qub_nqz"],
                        "gain": knobs["qub_gain"],
                        "freq": qubit_freq,
                    },
                    "readout": readout,
                },
                "relax_delay": knobs["relax_delay"],
                "reps": knobs["reps"],
                "rounds": knobs["rounds"],
                "sweep_range": sweep_range,
            },
            LenRabiCfgTemplate,
        )

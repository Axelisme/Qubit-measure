"""ro_optimize — 2D readout-optimisation Builder: argmax over freq × gain.

Sets this flux point's value on the picked flux device, runs a real readout
acquire over a freq × gain grid (against the flux-aware MockSoc offline or real
hardware), finds the optimum via ``argmax`` (no fit — the peak location IS the
result), fills its Sweep2DResult row in place, and returns a Patch with
``best_ro_freq`` and ``best_ro_gain``, plus the ``opt_readout`` module constructed
from them.

- requires the ``pi_pulse`` module (a pi-pulse is needed to prepare the excited
  state before measuring readout fidelity); placeholder default for the prototype.
- reads optional ``best_ro_freq`` and ``best_ro_gain`` (raw prev-point values —
  no smoothing flag: the tracking loop deliberately follows the actual last best
  to plant the Gaussian centre so the optimum tracks across flux points), with
  sensible MHz defaults when absent.
- reads optional ``t1`` (smoothed prev-point T1) for the relax_delay (3·T1) and
  to exercise the dependency mechanism.
- the ``readout`` module is optional (a base readout template); it is the readout
  the cfg sweeps over (its freq/gain are swept), mirroring the real experiment.

No fit step: the 2D landscape is computed in one shot (one effective "round"), so
``round_hook`` is called exactly once after filling the row.

``produce`` lowers the active context + this point's snapshot into a runnable
``RoOptimizeCfgTemplate`` via the Builder's ``make_cfg`` (Fast Fail if the context
is unconfigured — a real acquire needs a concrete ``pi_pulse`` + ``readout``). The
cfg's ``freq_range`` / ``gain_range`` (centred on the previous best, mirroring the
notebook ``RO_OptTask`` ``cfg_maker``) define the swept grid; the acquire runs the
real readout program over it. The mock path uses the flux-aware MockSoc, so the
fidelity landscape varies with the operating flux.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.utils import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    axis_to_sweep,
    parse_linear_axis,
    require_flux_device,
    set_flux_by_name,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Landscape2DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep2DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)

logger = logging.getLogger(__name__)


def _ro_signal2real(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Smooth the SNR landscape before argmax (lower-layer ``ro_opt_signal2real``)."""
    return np.abs(gaussian_filter(signals, sigma=1))


def _ro_landscape(
    tracker: MomentTracker, shape: tuple[int, ...]
) -> NDArray[np.float64]:
    """The smoothed (n_freq, n_gain) SNR landscape from the readout-channel tracker.

    The sweep is ``[("ge", 2), ("freq", n), ("gain", m)]`` plus a soft-average
    axis, so the tracker mean carries a leading reps singleton and the ge axis at
    index 1 — same layout the measure-side ``freq_gain`` ro_optimize reduces with
    ``ge_axis=1``. ``snr_as_signal`` reduces the ge axis to a per-(freq, gain) SNR;
    we reshape it to the Result's row shape (dropping the singleton) and smooth."""
    snr = snr_as_signal([tracker], ge_axis=1)
    return _ro_signal2real(np.asarray(snr, dtype=np.float64).reshape(shape))


# Default axis specs: (start, stop, npts)
_DEFAULT_FREQ: tuple[float, float, int] = (4998.0, 5002.0, 21)
_DEFAULT_GAIN: tuple[float, float, int] = (0.3, 0.7, 21)

_DEFAULT_CENTER_FREQ = 5000.0  # MHz — baseline readout resonance
_DEFAULT_CENTER_GAIN = 0.5

# the cfg sweep-window half-widths (the "設定頭"): the notebook centres the
# freq_range on the previous best ± ``0.2 * md.rf_w`` and the gain_range on the
# previous best ± ``0.05``. The GUI exposes those half-widths as params; these are
# their defaults when unset.
_DEFAULT_FREQ_WINDOW = 1.0  # MHz half-width of the readout-freq sweep window
_DEFAULT_GAIN_WINDOW = 0.05  # half-width of the readout-gain sweep window
_DEFAULT_T1 = 10.0  # us — fallback T1 for the relax_delay (3·T1)


class RoOptimizeModuleCfg(ConfigBase):
    """The modules ro_optimize lowers — an optional reset + the pi-pulse + readout.

    Mirrors the lower-layer ``experiment/v2/autofluxdep`` ``RO_OptModuleCfg``: the
    ``pi_pulse`` prepares the excited state and the ``readout`` is the pulse whose
    ``freq`` / ``gain`` the sweep optimises over.
    """

    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: PulseReadoutCfg


class RoOptimizeCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base program cfg ro_optimize lowers a context into.

    ``ProgramV2Cfg`` (reps/rounds/relax) + the ``ExpCfgModel`` device fields, plus
    the ``modules`` (pi_pulse + readout) and the ``freq_range`` / ``gain_range``
    sweep windows — exactly the lower-layer ``RO_OptCfgTemplate``. The flux ``dev``
    entry and the concrete ``freq`` / ``gain`` ``SweepCfg`` are merged in by the
    lower-layer ``run`` (the GUI prototype reads the window centres straight off
    the template to plant the synthetic landscape); ``freq_range`` / ``gain_range``
    are stripped before the runnable ``RO_OptCfg`` is validated downstream.
    """

    modules: RoOptimizeModuleCfg
    freq_range: tuple[float, float]
    gain_range: tuple[float, float]


def _default_t1() -> float:
    return _DEFAULT_T1


def _default_best_freq() -> float:
    return _DEFAULT_CENTER_FREQ


def _default_best_gain() -> float:
    return _DEFAULT_CENTER_GAIN


def _placeholder_pi_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real module
    return {"type": "pi", "length": 0.1}


def _default_readout() -> Any | None:
    return None


def _resolve_window(value: Any, default: float) -> float:
    """The half-width of a cfg sweep window from a param, or ``default`` if unset.

    The prototype's param fields are free text, so a missing / unparseable value
    degrades to the default rather than failing make_cfg."""
    if value is None or value == "":
        return default
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return default


class RoOptimizeNode(Node):
    """One flux point's ro_optimize: set flux → real acquire → SNR argmax → Patch.

    Mirrors the lower-layer ``RO_OptTask`` ``measure_ro_fn`` + ``run``: a
    ``ModularProgramV2`` (Reset → ge-Branch(pi_pulse) → PulseReadout) sweeps the
    readout freq × gain (interleaved with the ge axis), a ``MomentTracker``
    accumulates per-shot moments, and ``snr_as_signal`` turns them into an SNR
    landscape whose argmax is the best (freq, gain). No fit step.
    """

    def __init__(self, env: RunEnv, builder: RoOptimizeBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

        _ = snapshot["best_ro_freq"]  # optional — make_cfg centres the freq window
        _ = snapshot["best_ro_gain"]  # optional — make_cfg centres the gain window
        _ = snapshot["t1"]  # declared optional; relax_delay = 3·T1 in make_cfg
        _ = snapshot.module("pi_pulse")  # required — ge-branch excitation
        _ = snapshot.module("readout")  # required — the swept readout pulse

        result: Sweep2DResult = env.result
        freqs = result.freq
        gains = result.gain
        idx = env.flux_idx

        # Lower the active context into the run cfg (Fast Fail if unconfigured: a
        # real acquire needs a concrete pi_pulse + readout pulse).
        cfg = self._builder.make_cfg(env, snapshot)

        flux_device = require_flux_device(env, "ro_optimize")
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

        # Sweep the readout freq × gain over the Result's axes (lower layer sets the
        # freq / gain params on the readout pulse).
        freq_sweep = axis_to_sweep(freqs)
        gain_sweep = axis_to_sweep(gains)
        cfg.modules.readout.set_param("freq", sweep2param("freq", freq_sweep))
        cfg.modules.readout.set_param("gain", sweep2param("gain", gain_sweep))

        result.flux[idx] = env.flux

        tracker = MomentTracker()

        def on_round(_round_count: int, _avg_d: Any) -> None:
            # the SNR landscape (n_freq, n_gain) accumulated so far → overwrite row
            landscape = _ro_landscape(tracker, result.signal[idx].shape)
            np.copyto(result.signal[idx], landscape)
            if env.round_hook is not None:
                env.round_hook(idx)

        ModularProgramV2(
            env.soccfg,
            cfg,
            modules=[
                Reset("reset", cfg.modules.reset),
                Branch("ge", [], Pulse("pi_pulse", cfg.modules.pi_pulse)),
                PulseReadout("readout", cfg.modules.readout),
            ],
            sweep=[
                ("ge", 2),
                ("freq", freq_sweep),
                ("gain", gain_sweep),
            ],
        ).acquire(
            env.soc,
            progress=False,
            round_hook=on_round,
            trackers=[tracker],
        )

        landscape = _ro_landscape(tracker, result.signal[idx].shape)
        np.copyto(result.signal[idx], landscape)

        # argmax: project onto each axis and take the index of the max
        best_fi = int(np.argmax(landscape.max(axis=1)))
        best_gi = int(np.argmax(landscape.max(axis=0)))
        best_freq = float(freqs[best_fi])
        best_gain = float(gains[best_gi])

        result.best_freq[idx] = best_freq
        result.best_gain[idx] = best_gain

        logger.debug(
            "ro_optimize @flux%d: best_freq=%.3f best_gain=%.3f",
            idx,
            best_freq,
            best_gain,
        )

        # produce the tuned readout MODULE so downstream consumers (t1 / t2* / mist)
        # sweep against the optimised point: a deepcopy of the (real) readout cfg
        # with its freq / gain set to the argmax, mirroring the lower-layer
        # ``RO_OptTask.run`` (deepcopy(cfg.modules.readout); set_param freq/gain).
        opt_readout = deepcopy(cfg.modules.readout)
        opt_readout.set_param("freq", best_freq)
        opt_readout.set_param("gain", best_gain)

        patch = Patch()
        patch.set("best_ro_freq", best_freq)
        patch.set("best_ro_gain", best_gain)
        patch.set_module("opt_readout", opt_readout)
        return patch


class RoOptimizeBuilder(Builder):
    """The ro_optimize provider — 2D Gaussian synth, argmax (no fit), overwrite plot.

    Sweeps a freq × gain grid per flux point, synthesises a readout-fidelity
    landscape, and finds the optimum via argmax. No fitting: the Gaussian peak
    location IS the best readout point. Produces ``best_ro_freq``, ``best_ro_gain``,
    and the ``opt_readout`` module for downstream consumers (e.g. t1, mist).
    """

    name = "ro_optimize"
    provides = ("best_ro_freq", "best_ro_gain")
    provides_modules = ("opt_readout",)
    optional = (
        Dependency("t1", smooth="ewma", default=_default_t1),
        Dependency("best_ro_freq", default=_default_best_freq),
        Dependency("best_ro_gain", default=_default_best_gain),
    )
    requires_modules = (ModuleDep("pi_pulse", default=_placeholder_pi_pulse),)
    optional_modules = (ModuleDep("readout", default=_default_readout),)
    base_params = (
        "freq_expts",
        "gain_expts",
        "reps",
        "rounds",
        # the cfg sweep-window half-widths the cfg builder lowers into
        # freq_range / gain_range (centred on the previous best). The freq /
        # gain centres come from the snapshot; the rest from these params.
        "freq_window",
        "gain_window",
    )

    def make_init_result(self, params: Mapping[str, Any], flux: Any) -> Sweep2DResult:
        freq_expts: int = int(params.get("freq_expts") or _DEFAULT_FREQ[2])
        gain_expts: int = int(params.get("gain_expts") or _DEFAULT_GAIN[2])

        freqs = parse_linear_axis(
            params.get("freq_range"),
            (_DEFAULT_FREQ[0], _DEFAULT_FREQ[1], freq_expts),
        )
        gains = parse_linear_axis(
            params.get("gain_range"),
            (_DEFAULT_GAIN[0], _DEFAULT_GAIN[1], gain_expts),
        )
        return Sweep2DResult.allocate(flux, freqs, gains)

    def make_plotter(self, figure: Any) -> Landscape2DPlotter:
        return Landscape2DPlotter(figure, title="ro_optimize")

    def build_node(self, env: RunEnv) -> RoOptimizeNode:
        return RoOptimizeNode(env, self)

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> RoOptimizeCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the notebook's ro_optimize ``cfg_maker`` (runs in ``produce``,
        where the snapshot is available): the ``pi_pulse`` and ``readout`` modules
        come whole from the snapshot (lenrabi produces the pi-pulse; the readout is
        the base template), the relax_delay is ``3·T1`` (the smoothed prev-point
        T1), and the ``freq_range`` / ``gain_range`` are the previous best ± the
        window half-widths (the "設定頭"). The flux ``dev`` entry and the concrete
        ``freq`` / ``gain`` sweeps are NOT here — the lower-layer ``run`` merges
        them.

        Raises if the ml is unavailable or the pi_pulse / readout modules are
        unset — a real run needs both concrete modules (Fast Fail), unlike the
        synthetic path which fabricates a landscape.
        """
        params = env.params
        ml = env.ml
        if ml is None:
            raise RuntimeError("ro_optimize.make_cfg needs an active ModuleLibrary")
        pi_pulse = snapshot.module("pi_pulse")
        readout = snapshot.module("readout")
        if pi_pulse is None or readout is None:
            raise RuntimeError(
                "ro_optimize.make_cfg needs the pi_pulse + readout modules "
                "(none produced or preset)"
            )
        prev_best_freq = float(snapshot["best_ro_freq"])
        prev_best_gain = float(snapshot["best_ro_gain"])
        t1 = float(snapshot["t1"])
        freq_window = _resolve_window(params.get("freq_window"), _DEFAULT_FREQ_WINDOW)
        gain_window = _resolve_window(params.get("gain_window"), _DEFAULT_GAIN_WINDOW)
        return ml.make_cfg(
            {
                "modules": {
                    "pi_pulse": pi_pulse,
                    "readout": readout,
                },
                "relax_delay": 3.0 * t1,
                "reps": int(params.get("reps", 1000)),
                "rounds": int(params.get("rounds", 10)),
                "freq_range": (
                    prev_best_freq - freq_window,
                    prev_best_freq + freq_window,
                ),
                "gain_range": (
                    max(0.0, prev_best_gain - gain_window),
                    min(1.0, prev_best_gain + gain_window),
                ),
            },
            RoOptimizeCfgTemplate,
        )

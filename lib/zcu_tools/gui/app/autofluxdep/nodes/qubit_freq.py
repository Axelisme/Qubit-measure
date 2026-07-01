"""qubit_freq — the worked Builder: acquire → fit → fill Result → liveplot.

The first end-to-end real-acquire measurement provider in the Builder model. It
translates the notebook's ``cfg_maker`` lambda + ``ctx.env`` walrus chain into:

- a stateless ``QubitFreqBuilder`` declaring provides/requires + the sweep-lived
  factories (``make_init_result`` allocates the (n_flux, n_detune) Result;
  ``make_plotter`` builds the accumulating colormap Plotter);
- a short-lived ``QubitFreqNode`` (built per flux point by ``build_node``, with
  the flux point + soc + Result + round_hook curried in) whose ``produce``
  sets this point's flux on the picked flux device, recenters the detune sweep on
  the predicted qubit freq, runs a real ``TwoToneProgram.acquire`` (against the
  flux-aware MockSoc offline or real hardware), fits it with ``fit_qubit_freq``,
  fills the Result's flux-idx row in place, notifies via the round_hook, and
  returns the raw qubit_freq / fit_detune / fit_kappa Patch.

Mirrors the lower-layer ground truth ``experiment/v2/autofluxdep/qubit_freq.py``
(``QubitFreqTask.run`` + ``measure_fn``): predict-centred drive freq, detune
sweep via ``sweep2param``, ``setup_devices`` to push the flux, ``.acquire`` with
``round_hook`` + ``stop_checkers`` (cooperative stop + SNR early-stop), and the
predictor calibration closed loop.

- ``predict_freq`` — required; provided by the predictor Service (a Builder
  whose Node computes it), resolved latest-available like any dependency.
- ``fit_kappa`` — declared ``smooth="ewma"``: ``produce`` reads the *smoothed*
  estimate under the same key (the old ``qfw_factor``) and reports raw fit_kappa
  back; the orchestrator's SmoothingService projects the smoothed value in.
- ``readout`` — optional module, Node-produced (ro_optimize) → ml preset →
  default.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

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
    build_stop_checkers,
    is_good_fit,
    make_on_round,
    require_flux_device,
    round_progress,
    set_flux_by_name,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import READOUT_LIBRARY_ALIASES
from zcu_tools.gui.app.autofluxdep.nodes.plotters import title_with_snr
from zcu_tools.gui.app.autofluxdep.nodes.result import QubitFreqResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.waveform_defaults import waveform_or_const
from zcu_tools.program.v2 import SweepCfg, TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import rotate2real

logger = logging.getLogger(__name__)

_DEFAULT_QUB_WAVEFORM = "qub_flat"
_DEFAULT_QUB_CH = 0
_DEFAULT_EARLYSTOP_SNR = 50.0


class QubitFreqCfgTemplate(TwoToneCfg, ExpCfgModel):
    """The base two-tone cfg qubit_freq lowers a context into.

    Just ``TwoToneCfg`` (reps/rounds/relax + qub_pulse/readout modules) + the
    ``ExpCfgModel`` device/save fields — the flux ``dev`` entry and the ``detune``
    sweep are merged in by ``produce`` (the sweep recenters on the predicted freq,
    and the dev carries this flux point's value), mirroring the lower-layer
    ``experiment/v2/autofluxdep`` QubitFreqCfgTemplate.
    """


# --- default external bindings (project metadata can override) ---
def _default_kappa() -> float:
    # notebook: md.qf_w — the smoothed kappa estimate's fallback; lazy so the
    # real md is bound at build time. (Drives the drive-gain guess.)
    return 0.05


def _default_readout() -> Any | None:
    # last-resort readout if neither a Node produced one nor ml has a preset.
    return None


def _signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """PCA-rotate to the real axis and normalise to [0, 1] (a dip near 0)."""
    real = rotate2real(signals.astype(np.complex128)).real
    lo, hi = float(real.min()), float(real.max())
    real = (real - lo) / (hi - lo + 1e-12)
    # orient so the resonance is a dip (start/end high, centre low)
    if real[0] + real[-1] < real[len(real) // 2]:
        real = 1.0 - real
    return real


def detune_axis_to_sweep(detune: NDArray[np.float64]) -> SweepCfg:
    """Turn the Result's detune axis into the ``SweepCfg`` ``sweep2param`` needs.

    The Result stores the detune axis as an explicit array (from the lowered
    ``detune_sweep`` SweepCfg via ``sweepcfg_to_axis``); the program-side sweep is
    a ``SweepCfg`` (start/stop/expts/step). Reconstructs it from the axis endpoints
    + length so the FPGA sweep matches the Result's columns exactly."""
    det = np.asarray(detune, dtype=np.float64)
    expts = int(det.shape[0])
    start = float(det[0])
    stop = float(det[-1])
    step = 0.0 if expts == 1 else (stop - start) / (expts - 1)
    return SweepCfg(start=start, stop=stop, expts=expts, step=step)


class QubitFreqNode(Node):
    """One flux point's qubit_freq execution, environment curried in by build_node.

    Sets this point's flux on the picked flux device, recenters the detune sweep
    on the snapshot's ``predict_freq``, runs a real ``TwoToneProgram.acquire``
    against the connected board (the flux-aware MockSoc offline, or real hardware),
    fits the dip with ``fit_qubit_freq``, fills the sweep Result's ``flux_idx`` row
    in place, fires the ``round_hook`` so the main thread redraws, and returns the
    raw fit Patch.
    """

    def __init__(self, env: RunEnv, builder: QubitFreqBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        pred_qf = float(snapshot["predict_freq"])
        _ = snapshot["fit_kappa"]  # smoothed kappa (drives the real drive-gain)

        result: QubitFreqResult = env.result
        idx = env.flux_idx
        detunes = result.detune  # MHz, relative to the drive centre

        # Build the run cfg from the active context (Fast Fail if unconfigured: a
        # real acquire needs a concrete readout + drive pulse). The drive centre is
        # the predicted qubit freq (make_cfg sets qub_pulse.freq = predict_freq).
        cfg = self._builder.make_cfg(env, snapshot)
        center = float(cfg.modules.qub_pulse.freq)
        freqs = center + detunes  # absolute frequency axis (for the fit + plot)

        # Point the flux device at this sweep point and push it to hardware (mock:
        # writes the FakeDevice value → SimEngine reads it live). Fast Fail if no
        # flux source is picked — a real flux sweep must drive a device.
        flux_device = require_flux_device(env, "qubit_freq")
        # cfg.dev is typed Mapping but make_cfg always populates it with a mutable
        # dict (GlobalDeviceManager.get_all_info); cast for the in-place name write.
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

        # Recenter the detune sweep on the predicted freq (mirrors the lower layer:
        # qub_pulse.freq + detune_param), so the FPGA sweeps freq across the detune
        # window around the drive centre.
        detune_sweep = detune_axis_to_sweep(detunes)
        detune_param = sweep2param("detune", detune_sweep)
        cfg.modules.qub_pulse.set_param(
            "freq", cfg.modules.qub_pulse.freq + detune_param
        )

        result.flux[idx] = env.flux
        result.predict_freq[idx] = pred_qf

        # Real multi-round acquire. round_hook fires per round with the running
        # average; we rotate it to real, overwrite the Result row, and notify so the
        # liveplot settles round by round. The SNR probe + stop poll are threaded
        # into stop_checkers (early-stop on good SNR; cooperative cancel).
        probe = SnrProbe()
        stop_checkers = build_stop_checkers(env, probe, _signal2real)
        with round_progress(cfg.rounds, "qubit_freq", idx) as update_round_progress:
            on_round = make_on_round(
                result,
                idx,
                _signal2real,
                env.round_hook,
                probe=probe,
                round_progress_hook=update_round_progress,
            )
            raw = TwoToneProgram(
                env.soccfg, cfg, sweep=[("detune", detune_sweep)]
            ).acquire(
                env.soc,
                progress=False,
                round_hook=on_round,
                stop_checkers=stop_checkers,
            )
        real = _signal2real(acquire_to_complex(raw))

        # fit the fully-averaged signal
        freq, _, fwhm, _, fit_curve, _ = fit_qubit_freq(freqs, real)

        # fit-quality gate (the runner module's mean_err vs ptp): a poor fit is
        # discarded — it does NOT enter the Result, does NOT feed the predictor,
        # and is omitted from the Patch so downstream falls back to the latest
        # good value (and the next point predicts from the last good calibration).
        if not is_good_fit(real, fit_curve):
            logger.debug(
                "qubit_freq fit @flux%d: poor fit (SNR-trough?) — "
                "discarded, no calibrate, no qubit_freq produced",
                idx,
            )
            if env.round_hook is not None:
                env.round_hook(idx)  # raw row already shown; fit fields stay nan
            return Patch()  # partial: omit qubit_freq → downstream latest-available

        # good fit: fill the Result's fit fields + feed the closed-loop feedback
        result.fit_freq[idx] = float(freq)
        np.copyto(result.fit_curve[idx], np.asarray(fit_curve, dtype=np.float64))
        if env.round_hook is not None:
            env.round_hook(idx)

        # CLOSED-LOOP FEEDBACK: hand the measured frequency to the predictor so
        # the next flux point's predict_freq adapts (bias + IDW). qubit_freq
        # triggers calibration; it never touches the predictor's internals.
        if env.tools is not None and env.tools.predictor is not None:
            env.tools.predictor.calibrate(env.flux, float(freq))

        logger.debug(
            "qubit_freq fit @flux%d: freq=%.3f (predict=%.3f, detune=%+.3f) kappa=%.3f"
            " → calibrated predictor",
            idx,
            float(freq),
            pred_qf,
            float(freq) - pred_qf,
            float(fwhm),
        )

        patch = Patch()
        patch.set("qubit_freq", float(freq))
        patch.set("fit_detune", float(freq) - pred_qf)
        patch.set("fit_kappa", float(fwhm))
        return patch


class QubitFreqPlotter:
    """qubit_freq's two-panel liveplot, aligned with the runner module.

    Built once at Run start with a bare matplotlib ``Figure``; reuses
    ``zcu_tools.liveplot`` (LivePlot1D / LivePlot2DwithLine) embedded into the
    Figure's axes via ``existed_axes`` (the liveplot fig is None then — the host
    refreshes; see ``zcu_tools.liveplot.segments.base``). ``update(result, idx)``
    on the main thread after each row notification feeds:

    - ``fit_freq`` (LivePlot1D): flux value → fitted absolute qubit frequency.
    - ``detune`` (LivePlot2DwithLine): the flux × detune signal colormap plus the
      latest few flux rows as 1-D traces; a red dashed line marks the current
      fit_detune (matching the runner's ``freq_line``).
    """

    def __init__(self, figure: Any) -> None:
        from zcu_tools.liveplot import LivePlot1D, LivePlot2DwithLine

        self._fig = figure
        ax_fit = figure.add_subplot(2, 1, 1)
        ax_2d = figure.add_subplot(2, 2, 3)
        ax_line = figure.add_subplot(2, 2, 4)
        self._detune_title = "qubit_freq (detune)"
        self._freq_line = ax_line.axvline(np.nan, color="red", linestyle="--")
        self._fit = LivePlot1D(
            "Flux device value",
            "Frequency (MHz)",
            existed_axes=[[ax_fit]],
            segment_kwargs=dict(title="qubit_freq (fit_freq)"),
        )
        self._detune = LivePlot2DwithLine(
            "Flux device value",
            "Detune (MHz)",
            line_axis=1,
            num_lines=3,
            title=self._detune_title,
            existed_axes=[[ax_2d, ax_line]],
        )
        self._fit.__enter__()
        self._detune.__enter__()

    def update(self, result: QubitFreqResult, idx: int) -> None:
        self._fit.update(result.flux, result.fit_freq, refresh=False)
        self._detune.update(
            result.flux,
            result.detune,
            result.signal,
            title=title_with_snr(self._detune_title, result, idx),
            refresh=False,
        )
        # mark the current fit as a detune offset (freq - predict_freq)
        offset = result.fit_freq - result.predict_freq
        valid = offset[~np.isnan(offset)]
        self._freq_line.set_xdata([valid[-1] if valid.size else np.nan])
        self._fig.canvas.draw_idle()


class QubitFreqBuilder(Builder):
    """The qubit_freq provider — stateless; builds Result / Plotter / Nodes.

    Reports RAW fit results only (qubit_freq, fit_detune, fit_kappa) and reads
    the readout MODULE. A consumer wanting smoothed kappa adds ``smooth="ewma"``
    to its fit_kappa dependency; this provider never smooths its output.
    """

    name = "qubit_freq"
    provides = ("qubit_freq", "fit_detune", "fit_kappa")
    # predict_freq is required, supplied by the predictor Service (a Builder
    # whose Node computes it). With latest-available resolution a consumer
    # ordered before the predictor just reads the previous point's value.
    requires = (Dependency("predict_freq"),)
    optional = (
        # consumer-declared smoothing: read fit_kappa *smoothed* (same key). The
        # orchestrator builds the SmoothingService from this declaration alone.
        Dependency("fit_kappa", smooth="ewma", default=_default_kappa),
    )
    # the readout module: Node-produced → calibrated ml preset → default.
    optional_modules = (
        ModuleDep("readout", default=_default_readout, aliases=READOUT_LIBRARY_ALIASES),
    )

    def make_default_schema(self) -> NodeCfgSchema:
        """The typed node-knob schema (defaults + types) — the param SSOT.

        ``detune_sweep`` is a ``SweepSpec`` (expts-defined, aligning with the cfg
        editor): its default ``(-20, 50, expts=141)`` gives a 0.5 MHz step. The
        drive defaults follow the
        measure-gui qubit-pulse convention: ``qub_flat`` on channel 0, with
        ``qub_length`` at 0.1 us. If a project does not define that named
        waveform, ``make_cfg`` uses an inline const waveform of that length so
        mock-created contexts remain runnable.
        """
        return sectioned_node_schema(
            (
                node_section(
                    "sweep",
                    "Sweep",
                    node_field(
                        "detune_sweep",
                        "detune",
                        SweepSpec(label="Detune sweep (MHz)"),
                        SweepValue(start=-20.0, stop=50.0, expts=141),
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
                        100,
                    ),
                    node_field(
                        "relax_delay",
                        "relax_delay",
                        FloatSpec(label="Relax delay (us)"),
                        0.5,
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
    ) -> QubitFreqResult:
        knobs = schema.lower(None, md=md)
        detune = sweepcfg_to_axis(knobs["detune_sweep"])
        return QubitFreqResult.allocate(flux, detune)

    def make_plotter(self, figure: Any) -> QubitFreqPlotter:
        return QubitFreqPlotter(figure)

    def build_node(self, env: RunEnv) -> QubitFreqNode:
        return QubitFreqNode(env, self)

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> QubitFreqCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the notebook's qubit_freq ``cfg_maker`` (D1: runs in ``produce``,
        where the snapshot is available): the drive pulse frequency is the
        predicted qubit freq, the readout is the latest-available readout module,
        and the pulse waveform / channel / gain / nqz come from the node's params
        (the "設定頭"). The flux ``dev`` entry and the ``detune`` sweep are NOT here
        — ``produce`` merges them (the dev with this point's flux value, the sweep
        recentred on the predicted freq).

        Raises if the readout module is unavailable or the drive params are unset
        — a real run needs a concrete drive pulse (Fast Fail).
        """
        ml = env.ml
        if ml is None:
            raise RuntimeError("qubit_freq.make_cfg needs an active ModuleLibrary")
        readout = snapshot.module("readout")
        if readout is None:
            raise RuntimeError(
                "qubit_freq.make_cfg needs a readout module (none produced or preset)"
            )
        # the typed knobs (defaults + types owned by the placement's schema);
        # reps/rounds/relax come pre-typed, the optional drive waveform/ch are
        # omitted when unset.
        knobs = env.schema.lower(ml, md=env.md)
        waveform_name = knobs.get("qub_waveform")
        ch = knobs.get("qub_ch")
        if ch is None:
            raise RuntimeError("qubit_freq.make_cfg needs qub_ch param set")
        predict_freq = float(snapshot["predict_freq"])
        return ml.make_cfg(
            {
                "modules": {
                    "qub_pulse": {
                        "type": "pulse",
                        "waveform": waveform_or_const(
                            ml, waveform_name, length=knobs["qub_length"]
                        ),
                        "ch": ch,
                        "nqz": knobs["qub_nqz"],
                        "gain": knobs["qub_gain"],
                        "freq": predict_freq,
                    },
                    "readout": readout,
                },
                "relax_delay": knobs["relax_delay"],
                "reps": knobs["reps"],
                "rounds": knobs["rounds"],
            },
            QubitFreqCfgTemplate,
        )

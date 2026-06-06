"""qubit_freq — the worked Builder: synthesise → fit → fill Result → liveplot.

The first end-to-end measurement provider in the Builder model. It translates
the notebook's ``cfg_maker`` lambda + ``ctx.env`` walrus chain into:

- a stateless ``QubitFreqBuilder`` declaring provides/requires + the sweep-lived
  factories (``make_init_result`` allocates the (n_flux, n_detune) Result;
  ``make_plotter`` builds the accumulating colormap Plotter);
- a short-lived ``QubitFreqNode`` (built per flux point by ``build_node``, with
  the flux point + soc + Result + round_hook curried in) whose ``produce``
  synthesises a Lorentzian dip vs detune, fits it with the real
  ``fit_qubit_freq``, fills the Result's flux-idx row in place, notifies via the
  round_hook, and returns the raw qubit_freq / fit_detune / fit_kappa Patch.

Compare ``notebook_md/autofluxdep.md`` (the QubitFreqTask block):

    cfg_maker=lambda ctx, ml: (
        (info := ctx.env["info"])
        and (pred_qf := info["predict_freq"])                          # required
        and (prev_factor := info.last.get("qfw_factor", md.qf_w/0.05)) # smoothed kappa
        and (opt_readout := info.last.get("opt_readout", readout_cfg)) # optional module
        and ml.make_cfg({"modules": {"qub_pulse": {"gain": min(1.0, 6.5/prev_factor),
                                                    "freq": pred_qf},
                                     "readout": opt_readout}, ...}) )

- ``predict_freq`` — required; provided by the predictor Service (a Builder
  whose Node computes it), resolved latest-available like any dependency.
- ``fit_kappa`` — declared ``smooth="ewma"``: ``produce`` reads the *smoothed*
  estimate under the same key (the old ``qfw_factor``) and reports raw fit_kappa
  back; the orchestrator's SmoothingService projects the smoothed value in.
- ``readout`` — optional module, Node-produced (ro_optimize) → ml preset →
  default.

MockSoc.acquire returns only noise, so ``produce`` synthesises a
physically-plausible signal (``synth.lorentzian_dip``) and fits it with the real
fitter — exercising the whole acquire→fit→fill→notify→draw path without
hardware. Phase B swaps the synthesis for ``soc.acquire``.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Mapping, Optional

from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.result import QubitFreqResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    accumulate_rounds,
    lorentzian_dip,
    resolve_acquire_delay,
    resolve_rounds,
)
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import rotate2real

logger = logging.getLogger(__name__)

_DEFAULT_DETUNE = (-20.0, 50.0, 0.5)  # MHz: start, stop, step


# --- placeholder external bindings (Phase B: inject from project/metadata) ---
def _default_kappa() -> float:
    # notebook: md.qf_w — the smoothed kappa estimate's fallback; lazy so the
    # real md is bound at build time. (Drives the drive-gain guess.)
    return 0.05


def _default_readout() -> Optional[Any]:
    # last-resort readout if neither a Node produced one nor ml has a preset.
    return None


def parse_detune_sweep(spec: Any) -> NDArray[np.float64]:
    """Parse the detune sweep (text "start,stop,step" or a tuple) into an axis.

    Falls back to (-20, 50, 0.5) MHz if unset/unparseable — the prototype's
    field is free text, so a malformed value degrades to the default rather than
    failing the sweep."""
    try:
        if isinstance(spec, str) and spec.strip():
            start, stop, step = (float(x) for x in spec.split(","))
        elif isinstance(spec, (tuple, list)) and len(spec) == 3:
            start, stop, step = (float(x) for x in spec)
        else:
            start, stop, step = _DEFAULT_DETUNE
    except (ValueError, TypeError):
        start, stop, step = _DEFAULT_DETUNE
    n = max(2, int(round((stop - start) / step)) + 1)
    return np.linspace(start, stop, n)


def _signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """PCA-rotate to the real axis and normalise to [0, 1] (a dip near 0)."""
    real = rotate2real(signals.astype(np.complex128)).real
    lo, hi = float(real.min()), float(real.max())
    real = (real - lo) / (hi - lo + 1e-12)
    # orient so the resonance is a dip (start/end high, centre low)
    if real[0] + real[-1] < real[len(real) // 2]:
        real = 1.0 - real
    return real


class QubitFreqNode(Node):
    """One flux point's qubit_freq execution, environment curried in by build_node.

    Synthesises a Lorentzian dip vs detune (centred on the snapshot's
    ``predict_freq``), fits it, fills the sweep Result's ``flux_idx`` row in
    place, fires the ``round_hook`` so the main thread redraws, and returns the
    raw fit Patch. ``soc`` is the connected (mock) board, unused here — the
    signal is synthesised because MockSoc gives only noise.
    """

    def __init__(self, env: RunEnv) -> None:
        self._env = env

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        pred_qf = float(snapshot["predict_freq"])
        # fit_kappa is read smoothed (declared smooth="ewma") and the readout
        # module latest-available — both required for the real drive-gain guess /
        # readout cfg Phase B builds; the synthetic path here doesn't use them.
        _ = snapshot["fit_kappa"], snapshot.module("readout")

        result: QubitFreqResult = env.result
        detunes = result.detune
        freqs = pred_qf + detunes  # absolute frequency axis

        # plant a true resonance slightly off the prediction, realistic width.
        true_freq = pred_qf + 1.5  # MHz offset from prediction
        true_fwhm = 2.0  # MHz

        idx = env.flux_idx
        result.flux[idx] = env.flux
        result.predict_freq[idx] = pred_qf

        # emulate a multi-round acquire: each round is a fresh noise realisation,
        # the running average settles round by round (the row grows clearer as the
        # acquire progresses), and each round overwrites the row + notifies so the
        # liveplot shows the row converging. The total delay is split over rounds.
        base = env.flux_idx * 1000

        def make_round(k: int) -> NDArray[np.complex128]:
            return lorentzian_dip(freqs, true_freq, true_fwhm, seed=base + k)

        def on_round(avg: NDArray[np.complex128], _k: int) -> None:
            np.copyto(result.signal[idx], _signal2real(avg))
            if env.round_hook is not None:
                env.round_hook(idx)

        averaged = accumulate_rounds(
            make_round,
            resolve_rounds(env.params),
            on_round,
            delay=resolve_acquire_delay(env.params),
        )
        real = _signal2real(averaged)

        # fit the fully-averaged signal, then fill the fit fields + notify again
        freq, _, fwhm, _, fit_curve, _ = fit_qubit_freq(freqs, real)
        result.fit_freq[idx] = float(freq)
        np.copyto(result.fit_curve[idx], np.asarray(fit_curve, dtype=np.float64))
        if env.round_hook is not None:
            env.round_hook(idx)

        logger.debug(
            "qubit_freq fit @flux%d: freq=%.3f (predict=%.3f, detune=%+.3f) kappa=%.3f",
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
            title="qubit_freq (detune)",
            existed_axes=[[ax_2d, ax_line]],
        )
        self._fit.__enter__()
        self._detune.__enter__()

    def update(self, result: QubitFreqResult, idx: int) -> None:
        del idx  # the whole accumulated map is redrawn; idx is just the trigger
        self._fit.update(result.flux, result.fit_freq, refresh=False)
        self._detune.update(result.flux, result.detune, result.signal, refresh=False)
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
    # the readout module: Node-produced (ro_optimize) → ml preset → default.
    optional_modules = (ModuleDep("readout", default=_default_readout),)
    base_params = (
        "detune_sweep",
        "reps",
        "rounds",
        "relax_delay",
        "earlystop_snr",
        "acquire_delay",
    )

    def make_init_result(self, params: Mapping[str, Any], flux: Any) -> QubitFreqResult:
        detune = parse_detune_sweep(params.get("detune_sweep"))
        return QubitFreqResult.allocate(flux, detune)

    def make_plotter(self, figure: Any) -> QubitFreqPlotter:
        return QubitFreqPlotter(figure)

    def build_node(self, env: RunEnv) -> QubitFreqNode:
        return QubitFreqNode(env)

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
from zcu_tools.gui.app.autofluxdep.nodes.synth import lorentzian_dip
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
        signals = lorentzian_dip(freqs, true_freq, true_fwhm, seed=env.flux_idx)
        real = _signal2real(signals)

        # fill the raw row first (a mid-acquire row has signal, nan fit) + notify
        idx = env.flux_idx
        result.flux[idx] = env.flux
        result.predict_freq[idx] = pred_qf
        np.copyto(result.signal[idx], real)
        if env.round_hook is not None:
            env.round_hook(idx)

        # fit, then fill the fit fields + notify again (fit curve now present)
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
    """Accumulating flux×detune colormap, redrawn each flux point on the main thread.

    Built once at Run start (lifetime = whole sweep) with a bare matplotlib
    ``Figure``. ``update(result, idx)`` is called on the main thread after each
    row-updated notification: it redraws the whole accumulated colormap (every
    settled flux row plus the current one) and overlays the per-row fitted freq
    as a tracking line. Holds drawing state but never owns the Qt widget.
    """

    def __init__(self, figure: Any) -> None:
        self._fig = figure
        self._ax = figure.add_subplot(111)
        self._im = None
        self._fit_line = None
        self._ax.set_xlabel("detune (MHz)")
        self._ax.set_ylabel("flux index")
        self._ax.set_title("qubit_freq")

    def update(self, result: QubitFreqResult, idx: int) -> None:
        del idx  # the whole accumulated map is redrawn; idx is just the trigger
        det = result.detune
        extent = (float(det[0]), float(det[-1]), -0.5, result.n_flux - 0.5)
        if self._im is None:
            self._im = self._ax.imshow(
                result.signal,
                aspect="auto",
                origin="lower",
                extent=extent,
                interpolation="nearest",
            )
        else:
            self._im.set_data(result.signal)
            self._im.autoscale()

        # overlay the fitted qubit freq as a detune offset (freq - predict_freq)
        fit_detune = result.fit_freq - result.predict_freq
        rows = np.arange(result.n_flux, dtype=np.float64)
        if self._fit_line is None:
            (self._fit_line,) = self._ax.plot(
                fit_detune, rows, "r.-", linewidth=1.0, markersize=3
            )
        else:
            self._fit_line.set_data(fit_detune, rows)
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
    base_params = ("detune_sweep", "reps", "rounds", "relax_delay", "earlystop_snr")

    def make_init_result(
        self, params: Mapping[str, Any], n_flux: int
    ) -> QubitFreqResult:
        detune = parse_detune_sweep(params.get("detune_sweep"))
        return QubitFreqResult.allocate(n_flux, detune)

    def make_plotter(self, figure: Any) -> QubitFreqPlotter:
        return QubitFreqPlotter(figure)

    def build_node(self, env: RunEnv) -> QubitFreqNode:
        return QubitFreqNode(env)

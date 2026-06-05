"""qubit_freq Node — the worked example translating the notebook's cfg_maker.

Shows how one measurement's runner-based ``cfg_maker`` lambda + ``ctx.env``
walrus chain becomes a declarative ``NodeSpec`` + a thin ``build_cfg``. Compare
``notebook_md/autofluxdep.md`` (the QubitFreqTask block) line-by-line:

    cfg_maker=lambda ctx, ml: (
        (info := ctx.env["info"])
        and (pred_qf := info["predict_freq"])                          # required
        and (prev_factor := info.last.get("qfw_factor", md.qf_w/0.05)) # smoothed kappa
        and (opt_readout := info.last.get("opt_readout", readout_cfg)) # optional
        and ml.make_cfg({
            "modules": {"qub_pulse": {"gain": min(1.0, 6.5/prev_factor),  # (3) arithmetic
                                       "freq": pred_qf},
                        "readout": opt_readout}, ...}, QubitFreqCfgTemplate)
    )

- ``predict_freq`` — required; seeded each point by the per-point pre-step (the
  notebook's before_each). With latest-available resolution a key not produced
  by any Node just falls back; here it is always seeded so ``requires`` is fine.
- ``fit_kappa`` — declared ``smooth="ewma"``, so the Node reads the *smoothed*
  estimate under the same key (the old ``qfw_factor``). It reports raw fit_kappa
  and the orchestrator's SmoothingService projects the smoothed value back in.
- ``opt_readout`` — optional with a default.
- The ``min(1.0, ...)`` arithmetic is all that remains in ``build_cfg`` — pure,
  no walrus, no ``.get``, no time scope.

The defaults below are placeholders; Phase B injects the real external bindings
(``md.qf_w``, the project's ``readout_cfg``) via a factory so they aren't
captured at import time.
"""

from __future__ import annotations

from typing_extensions import Any, Mapping, Optional

from zcu_tools.gui.app.autofluxdep.nodes.spec import (
    Dependency,
    ModuleDep,
    NodeSpec,
)


# --- placeholder external bindings (Phase B: inject from project/metadata) ---
def _default_kappa() -> float:
    # notebook: md.qf_w — the smoothed kappa estimate's fallback; lazy so the
    # real md is bound at build time. (Drives the drive-gain guess.)
    return 0.05


def _default_readout() -> Optional[Any]:
    # last-resort readout if neither a Node produced one nor ml has a preset.
    return None


def _build_cfg(snapshot: Any, params: Mapping[str, Any], tools: Any) -> Optional[Any]:
    """The surviving (3) arithmetic step. Returns a cfg dict shape (skeleton).

    No hardware, no real cfg model yet — returns a plain dict mirroring the
    notebook's make_cfg payload so the dependency wiring is demonstrable.
    Phase B returns a validated ``QubitFreqCfg``.

    Reads its resolved snapshot:
    - ``snapshot["predict_freq"]`` — seeded each point by the pre-step via
      ``tools.predictor.predict_freq``.
    - ``snapshot["fit_kappa"]``    — declared ``smooth="ewma"``, so this is the
      *smoothed* estimate (same key). The Node never knows it is smoothed.
    - ``snapshot.module("readout")`` — the readout MODULE, resolved
      latest-available: a Node-produced tuned readout (ro_optimize) if any, else
      the ml preset, else the declared default. Read like any module.

    ``tools`` (predictor) is available; the predictor *write* (update_bias) is
    Phase B's result post-processing.
    """
    _ = tools  # predictor read happens in the pre-step; write in post-step
    pred_qf = snapshot["predict_freq"]
    kappa = snapshot["fit_kappa"]  # smoothed (declared smooth="ewma"), same key
    readout = snapshot.module("readout")  # latest-available module

    return {
        "modules": {
            "qub_pulse": {
                # drive-gain guess from the smoothed kappa (pure math)
                "gain": min(1.0, kappa / 0.05 * 0.05),
                "freq": pred_qf,
            },
            "readout": readout,
        },
        "reps": params.get("reps", 1000),
        "rounds": params.get("rounds", 100),
        "relax_delay": params.get("relax_delay", 0.5),
        "sweep": {"detune": params.get("detune_sweep")},
    }


# The Node reports RAW fit results only — qubit_freq, fit_detune, fit_kappa — and
# reads the readout MODULE (which a tuning Node like ro_optimize may produce, or
# the ml library presets). A consumer that wants the smoothed kappa just adds
# smooth="ewma" to the fit_kappa dependency; the Node never smooths its output.
QUBIT_FREQ_SPEC = NodeSpec(
    name="qubit_freq",
    provides=("qubit_freq", "fit_detune", "fit_kappa"),
    # predict_freq is seeded by the pre-step each point; no Node provides it, but
    # with latest-available resolution a missing value just falls back (here it
    # is always seeded, so required is fine).
    requires=(Dependency("predict_freq"),),
    optional=(
        # consumer-declared smoothing: read fit_kappa *smoothed* (same key). The
        # orchestrator builds the SmoothingService from this declaration alone.
        Dependency("fit_kappa", smooth="ewma", default=_default_kappa),
    ),
    # the readout module: Node-produced (ro_optimize) → ml preset → default.
    optional_modules=(ModuleDep("readout", default=_default_readout),),
    base_params=("detune_sweep", "reps", "rounds", "relax_delay", "earlystop_snr"),
    build_cfg=_build_cfg,
)

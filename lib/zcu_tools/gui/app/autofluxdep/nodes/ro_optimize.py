"""ro_optimize Node — the worked example of a Node that PRODUCES a module.

In the notebook, ro_optimize sweeps readout freq/gain, picks the best, builds a
tuned readout cfg, and stashes it as ``info["opt_readout"]`` for downstream Nodes
(t1 / t2 / qubit_freq) to use. That tuned readout is really a *module*, not a
plain info value — so here it lives in the module space:

- ro_optimize declares ``provides_modules=("readout",)`` and emits the tuned
  readout via ``patch.set_module("readout", tuned)``.
- A downstream Node declaring ``ModuleDep("readout")`` then resolves it
  latest-available: the tuned module this/previous point, else the ml preset.

So the old ``opt_readout`` carried through info is gone; the readout module flows
through the module space under its plain name ``readout``, symmetric to how info
values flow. ro_optimize reads the *base* readout module to tune, and writes the
tuned one back under the same name.
"""

from __future__ import annotations

from typing_extensions import Any, Mapping, Optional

from zcu_tools.gui.app.autofluxdep.nodes.spec import (
    Dependency,
    ModuleDep,
    NodeSpec,
)


def _default_readout() -> Optional[Any]:
    return None


def _build_cfg(snapshot: Any, params: Mapping[str, Any], tools: Any) -> Optional[Any]:
    """Build the ro-optimization sweep cfg around the base readout module.

    Skeleton: returns a dict mirroring the sweep payload. The *tuning* (picking
    best freq/gain and producing the new readout module) happens in result
    post-processing — see the patch the Node returns at run time, which carries
    the tuned module via ``patch.set_module("readout", ...)``.
    """
    _ = tools
    base_readout = snapshot.module("readout")
    return {
        "modules": {"readout": base_readout},
        "reps": params.get("reps", 1000),
        "rounds": params.get("rounds", 10),
        "sweep": {
            "freq": params.get("freq_expts", 10),
            "gain": params.get("gain_expts", 10),
        },
    }


RO_OPTIMIZE_SPEC = NodeSpec(
    name="ro_optimize",
    provides=("best_ro_freq", "best_ro_gain"),
    provides_modules=("readout",),  # ← produces the tuned readout MODULE
    optional=(Dependency("smooth_t1", smooth="ewma", default=lambda: 1.0),),
    # reads the base readout to tune (ml preset, or a previously tuned one)
    optional_modules=(ModuleDep("readout", default=_default_readout),),
    base_params=("freq_expts", "gain_expts", "reps", "rounds"),
    build_cfg=_build_cfg,
)

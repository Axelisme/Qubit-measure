"""ProjectService — read the fluxonium fit inputs from ``params.json``.

The cross-app handoff: fluxdep-gui writes the ``fluxdep_fit`` section (EJ/EC/EL +
flux alignment), dispersive reads it as its hard inputs (notebook cell 2). The
``bare_rf`` seed is sourced by priority: a prior ``dispersive`` section's bare_rf
→ the fit's ``plot_transitions.r_f`` → the default (5.0 GHz).

Fast-fails clearly when ``fluxdep_fit`` is absent — it is the hard prerequisite,
so the message points the user at running fluxdep-gui first.
"""

from __future__ import annotations

import logging
import os

from zcu_tools.gui.app.dispersive.state import (
    DEFAULT_BARE_RF,
    DispersiveState,
    FluxoniumInputs,
)
from zcu_tools.notebook.persistance import load_result

logger = logging.getLogger(__name__)


def default_params_path(result_dir: str) -> str:
    """The conventional ``params.json`` location for a project (``<result_dir>``)."""
    return os.path.join(result_dir, "params.json")


class ProjectService:
    """Reads ``params.json`` into the State's fluxonium fit inputs."""

    def __init__(self, state: DispersiveState) -> None:
        self._state = state

    def load_fit_inputs(self, params_path: str | None = None) -> FluxoniumInputs:
        """Read the ``fluxdep_fit`` section and record it as the fit inputs.

        ``params_path`` defaults to ``<result_dir>/params.json``. Fast-fails if the
        file or its ``fluxdep_fit`` section is missing (the fluxdep step must have
        run first). Returns the parsed inputs (also written into State).
        """
        path = params_path or default_params_path(self._state.project.result_dir)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"params.json not found at {path!r} — run fluxdep-gui first to "
                "produce the fluxdep_fit section dispersive reads"
            )

        result = load_result(path)
        fit = result.get("fluxdep_fit")
        if fit is None:
            raise ValueError(
                f"{path!r} has no 'fluxdep_fit' section — run fluxdep-gui first to "
                "fit (EJ, EC, EL) and the flux alignment"
            )

        p = fit["params"]
        params = (float(p["EJ"]), float(p["EC"]), float(p["EL"]))
        bare_rf_seed = self._derive_bare_rf(result, fit)

        inputs = FluxoniumInputs(
            params=params,
            flux_half=float(fit["flux_half"]),
            flux_int=float(fit["flux_int"]),
            flux_period=float(fit["flux_period"]),
            bare_rf_seed=bare_rf_seed,
        )
        self._state.set_fit_inputs(inputs)
        logger.debug("load_fit_inputs: params=%s bare_rf_seed=%s", params, bare_rf_seed)
        return inputs

    @staticmethod
    def _derive_bare_rf(result, fit) -> float:
        """bare_rf priority: prior dispersive section → fit r_f → default."""
        dispersive = result.get("dispersive")
        if dispersive is not None and "bare_rf" in dispersive:
            return float(dispersive["bare_rf"])
        transitions = fit.get("plot_transitions") or {}
        if "r_f" in transitions:
            return float(transitions["r_f"])
        return DEFAULT_BARE_RF

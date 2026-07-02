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

from zcu_tools.gui.app.dispersive.state import (
    DEFAULT_BARE_RF,
    DispersiveState,
    FluxoniumInputs,
)
from zcu_tools.meta_tool import (
    QubitParams,
    QubitParamsError,
    params_path_for_result_dir,
)

logger = logging.getLogger(__name__)


def default_params_path(result_dir: str) -> str:
    """The conventional ``params.json`` location for a project (``<result_dir>``)."""
    return params_path_for_result_dir(result_dir)


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
        try:
            loaded = QubitParams(path, readonly=True).require_dispersive_inputs(
                default_bare_rf=DEFAULT_BARE_RF
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"params.json not found at {path!r} — run fluxdep-gui first to "
                "produce the fluxdep_fit section dispersive reads"
            ) from exc
        except QubitParamsError as exc:
            if exc.reason_code == "fluxdep_fit_missing":
                raise ValueError(
                    f"{path!r} has no 'fluxdep_fit' section — run fluxdep-gui first to "
                    "fit (EJ, EC, EL) and the flux alignment"
                ) from exc
            raise ValueError(
                f"Failed to read fluxdep_fit from {path!r}: {exc}"
            ) from exc

        inputs = FluxoniumInputs(
            params=loaded.params,
            flux_half=loaded.flux_half,
            flux_int=loaded.flux_int,
            flux_period=loaded.flux_period,
            bare_rf_seed=loaded.bare_rf_seed,
        )
        self._state.set_fit_inputs(inputs)
        logger.debug(
            "load_fit_inputs: params=%s bare_rf_seed=%s",
            loaded.params,
            loaded.bare_rf_seed,
        )
        return inputs

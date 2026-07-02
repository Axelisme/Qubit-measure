"""ExportService — write the fitted g / bare_rf back to ``params.json``."""

from __future__ import annotations

import logging

from zcu_tools.gui.app.dispersive.services.project import default_params_path
from zcu_tools.gui.app.dispersive.state import DispersiveState
from zcu_tools.meta_tool import DispersiveFit, QubitParams, QubitParamsError

logger = logging.getLogger(__name__)


class ExportService:
    """Writes the dispersive fit result into ``params.json``'s dispersive section."""

    def __init__(self, state: DispersiveState) -> None:
        self._state = state

    def export_params(self, savepath: str | None = None) -> str:
        """Write ``{g, bare_rf}`` into ``params.json``, preserving ``fluxdep_fit``.

        ``savepath`` defaults to ``<result_dir>/params.json``. Fast-fails when no fit
        result is recorded, or when the target file does not exist (the fluxdep_fit
        inputs must already be there). Returns the written path.
        """
        fit = self._state.disp_fit
        if not fit.has_result or fit.bare_rf is None:
            raise RuntimeError("no dispersive fit result to export (fit g first)")

        path = savepath or default_params_path(self._state.project.result_dir)
        assert fit.g is not None  # has_result guards this
        try:
            QubitParams(path).set_dispersive_fit(
                DispersiveFit(g=fit.g, bare_rf=fit.bare_rf)
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"params.json not found at {path!r} — it must already hold the "
                "fluxdep_fit section (load the fit inputs from it first)"
            ) from exc
        except QubitParamsError as exc:
            raise ValueError(
                f"Failed to export dispersive fit to {path!r}: {exc}"
            ) from exc
        logger.debug("export_params: wrote dispersive to %r", path)
        return path

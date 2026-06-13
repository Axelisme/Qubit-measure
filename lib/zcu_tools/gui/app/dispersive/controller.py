"""Controller — the dispersive-fit-gui façade.

Holds the State + EventBus and the domain services, and is the single command
surface for both Views (the Qt MainWindow and the read-only RemoteControlAdapter).
Services stay pure (they read/write State and bump versions, Qt-free,
independently testable); the Controller calls a service and then emits the
corresponding EventBus event so Views can react.

It has only the dispersive single-flow pipeline actions: load fit inputs from
params.json → load a one-tone spectrum → preprocess → manually tune g & bare_rf
(``set_manual_fit`` records the accepted values — there is no auto-fit) → export the
dispersive section back to params.json. ``predict_dispersive`` is a read for the
tuning figure (no State write, no event).
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from zcu_tools.gui.app.dispersive.event_bus import (
    DispFitChangedPayload,
    EventBus,
    FitInputsLoadedPayload,
    OnetoneLoadedPayload,
    PreprocessChangedPayload,
    ProjectChangedPayload,
)
from zcu_tools.gui.app.dispersive.services.export import ExportService
from zcu_tools.gui.app.dispersive.services.load import LoadService
from zcu_tools.gui.app.dispersive.services.predict import PredictService
from zcu_tools.gui.app.dispersive.services.preprocess import PreprocessService
from zcu_tools.gui.app.dispersive.services.project import ProjectService
from zcu_tools.gui.app.dispersive.state import (
    DispersiveState,
    PreprocessResult,
)
from zcu_tools.gui.controller_base import BaseController
from zcu_tools.gui.project import ProjectInfo

logger = logging.getLogger(__name__)


class Controller(BaseController[DispersiveState, EventBus]):
    """Command façade over the dispersive pipeline services."""

    def __init__(
        self,
        state: DispersiveState,
        bus: EventBus | None = None,
        project_root: str | None = None,
    ) -> None:
        super().__init__(state, bus if bus is not None else EventBus(), project_root)
        self._project = ProjectService(state)
        self._load = LoadService(state)
        self._preprocess = PreprocessService(state)
        self._export = ExportService(state)
        # PredictService is bound to one (params, flux-axis); rebuilt lazily when
        # the preprocessing result or fit inputs change (see _predictor).
        self._predict: PredictService | None = None
        self._predict_key: tuple | None = None

    # --- project ---------------------------------------------------------

    def setup_project(self, project: ProjectInfo) -> None:
        self._state.set_project(project)
        self._emit(ProjectChangedPayload())

    def load_fit_inputs(self, params_path: str | None = None) -> None:
        self._project.load_fit_inputs(params_path)
        self._invalidate_predictor()
        self._emit(FitInputsLoadedPayload(has_inputs=True))

    # --- load ------------------------------------------------------------

    def load_onetone(self, filepath: str, transpose_axes: bool = False) -> str:
        name = self._load.load_onetone(filepath, transpose_axes=transpose_axes)
        self._invalidate_predictor()
        self._emit(OnetoneLoadedPayload(name=name))
        return name

    # --- preprocess ------------------------------------------------------

    def compute_preprocess(self) -> PreprocessResult:
        """Run preprocessing — pure, off-main-safe (no State write, no event)."""
        return self._preprocess.compute()

    def record_preprocess(self, result: PreprocessResult) -> None:
        """Write a computed preprocessing result onto State (MAIN THREAD only)."""
        self._preprocess.record(result)
        self._invalidate_predictor()
        self._emit(PreprocessChangedPayload())

    def preprocess(self) -> PreprocessResult:
        """Compute + record inline (RPC / convenience path, main thread)."""
        result = self._preprocess.preprocess()
        self._invalidate_predictor()
        self._emit(PreprocessChangedPayload())
        return result

    # --- predict (live tuning, synchronous read) -------------------------

    def predict_dispersive(
        self,
        g: float,
        bare_rf: float,
        *,
        return_dim: int = 2,
    ) -> tuple[NDArray[np.float64], ...]:
        """LRU-cached dispersive prediction over the full preprocessed flux axis."""
        return self._predictor().predict(g, bare_rf, return_dim=return_dim)

    def predict_flux_axis(self) -> NDArray[np.float64]:
        """The preprocessed flux axis a ``predict_dispersive`` call runs over."""
        return self._predictor().flux_axis()

    def predict_sample_points(
        self,
        fluxs: NDArray[np.float64],
        g: float,
        bare_rf: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Ground/excited resonator frequencies (GHz) at arbitrary sample fluxs.

        The live single-point path for the draggable sample-flux lines (read-only,
        synchronous, cheap). Fast-fails if no fit inputs are loaded.
        """
        from zcu_tools.gui.app.dispersive.services.predict import predict_dispersive_at

        inputs = self._state.fit_inputs
        if inputs is None:
            raise RuntimeError("no fluxonium fit inputs (load params.json first)")
        rf_0, rf_1 = predict_dispersive_at(inputs.params, fluxs, g, bare_rf)
        return rf_0, rf_1

    def auto_tune(
        self,
        sample_fluxs: NDArray[np.float64],
        g0: float,
        bare_rf0: float,
        g_bounds: tuple[float, float],
        rf_bounds: tuple[float, float],
    ) -> tuple[float, float]:
        """Optimise (g, bare_rf) to best match the sample-flux lines (read-only).

        Snapshots the fit params + preprocessing off State, then runs the scipy
        optimiser. Pure and State-free, so the (slow, iterative) call is safe on a
        worker thread; the caller records the result on the main thread. Fast-fails
        if inputs / preprocessing are missing or there are no sample fluxes.
        """
        from zcu_tools.gui.app.dispersive.services.autotune import auto_tune

        inputs = self._state.fit_inputs
        pp = self._state.preprocess
        if inputs is None:
            raise RuntimeError("no fluxonium fit inputs (load params.json first)")
        if pp is None:
            raise RuntimeError("no preprocessing result (run preprocessing first)")
        return auto_tune(
            inputs.params,
            pp.sp_fluxs,
            pp.sp_freqs,
            pp.norm_phases,
            sample_fluxs,
            g0,
            bare_rf0,
            g_bounds,
            rf_bounds,
        )

    # --- result (the user's tuning is the final fit) --------------------

    def set_manual_fit(self, g: float, bare_rf: float, *, res_dim: int = 4) -> None:
        """Record the accepted g / bare_rf (+ its sim resolution) as the result."""
        self._state.set_disp_result(g, bare_rf, res_dim=res_dim)
        self._emit(DispFitChangedPayload(has_result=True))

    # --- export ----------------------------------------------------------

    def export_params(self, savepath: str | None = None) -> str:
        return self._export.export_params(savepath)

    # --- predictor lifecycle (cache bound to params + flux axis) ---------

    def _predictor(self) -> PredictService:
        """The PredictService for the current (params, flux-axis), built on demand.

        Rebuilt whenever the fluxonium params or the preprocessed flux axis change,
        so a stale cache can never serve a prediction for an old spectrum.
        """
        inputs = self._state.fit_inputs
        pp = self._state.preprocess
        if inputs is None:
            raise RuntimeError("no fluxonium fit inputs (load params.json first)")
        if pp is None:
            raise RuntimeError("no preprocessing result (run preprocessing first)")
        key = (inputs.params, id(pp))
        if self._predict is None or self._predict_key != key:
            self._predict = PredictService(inputs.params, pp.sp_fluxs)
            self._predict_key = key
        return self._predict

    def _invalidate_predictor(self) -> None:
        self._predict = None
        self._predict_key = None

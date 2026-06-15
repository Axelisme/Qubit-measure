"""PredictorService — FluxoniumPredictor loading and frequency prediction.

Owns the ``exp_context.predictor`` write seam (set_context + PredictorChangedPayload)
and all synchronous prediction computation. This is pure computation: no Qt signals,
no operation runner, no exclusion gate. Errors surface as PredictorLoadError /
PredictorNotLoaded for the caller to translate.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.gui.session.events import PredictorChangedPayload
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.state import SessionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadPredictorRequest:
    # path is Optional so the same request type can represent both the file-load
    # path (a concrete params.json) and the in-memory install path (None — the
    # predictor is built from typed params, no backing file). load_predictor
    # itself still requires a concrete path; None is rejected fast there.
    path: str | None
    flux_bias: float


@dataclass(frozen=True)
class SetModelParamsRequest:
    """Build a FluxoniumPredictor straight from typed model params (no file).

    EJ/EC/EL are the Fluxonium energies in GHz; flux_half/flux_period are the
    value->flux affine anchors (device-value units); flux_bias is the predictor
    bias correction. Routes through install_predictor (the in-memory seam) so a
    user/agent can plug in trial energies (e.g. EJ:EC:EL = 4:1:1) directly.
    """

    EJ: float
    EC: float
    EL: float
    flux_half: float
    flux_period: float
    flux_bias: float = 0.0


@dataclass(frozen=True)
class PredictFreqRequest:
    value: float
    transition: tuple[int, int]


@dataclass(frozen=True)
class PredictCurveRequest:
    """Request a batch of transition-frequency curves over a device-value grid."""

    values: NDArray[np.float64]
    transitions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PredictCurveResult:
    """Result of a batch curve computation."""

    labels: tuple[str, ...]
    values: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    # shape: (n_transitions, n_values), frequencies in MHz
    freqs_mhz: NDArray[np.float64]


@dataclass(frozen=True)
class PredictMatrixCurveRequest:
    """Request a batch of transition matrix-element magnitudes over a device-value grid."""

    values: NDArray[np.float64]
    transitions: tuple[tuple[int, int], ...]
    operator: Literal["n", "phi"]


@dataclass(frozen=True)
class PredictMatrixCurveResult:
    """Result of a batch matrix-element curve computation."""

    labels: tuple[str, ...]
    values: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    # shape: (n_transitions, n_values), dimensionless magnitudes |<i|op|j>|
    mags: NDArray[np.float64]


# ---------------------------------------------------------------------------
# Typed expected failures
# ---------------------------------------------------------------------------


class PredictorLoadError(RuntimeError):
    """Expected failure: predictor file could not be loaded / parsed."""


class PredictorNotLoaded(RuntimeError):
    """Expected failure: predict_freq called before any predictor was loaded."""


# ---------------------------------------------------------------------------
# Pure query: read fluxdep_fit model params from a params.json
# ---------------------------------------------------------------------------


def read_fluxdep_fit_params(path: str) -> SetModelParamsRequest:
    """Read a params.json and return its fluxdep_fit model params.

    Pure query (no state/bus) for the dialog's "Load -> fields" action: it parses
    the file but does NOT install a predictor, so the user can tweak the populated
    fields before applying. flux_bias defaults to 0.0 (the file's fluxdep_fit
    carries no bias — that is a per-measurement correction). flux_int is read by
    the fit but deliberately omitted here: it is alignment-only, not part of the
    predictor's value<->flux affine.

    Reuses load_result (the same loader FluxoniumPredictor.from_file uses).
    Raises PredictorLoadError on any IO / missing-key problem.
    """
    from zcu_tools.notebook.persistance import load_result

    try:
        result = load_result(path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise PredictorLoadError(f"Failed to read params file: {exc}") from exc
    fluxdep_fit = result.get("fluxdep_fit")
    if fluxdep_fit is None:
        raise PredictorLoadError("params file has no 'fluxdep_fit' section")
    try:
        params = fluxdep_fit["params"]
        return SetModelParamsRequest(
            EJ=params["EJ"],
            EC=params["EC"],
            EL=params["EL"],
            flux_half=fluxdep_fit["flux_half"],
            flux_period=fluxdep_fit["flux_period"],
        )
    except KeyError as exc:
        raise PredictorLoadError(f"fluxdep_fit is missing key {exc}") from exc


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class PredictorService:
    """Owns the exp_context.predictor write seam and all prediction computation.

    Pure class (not QObject): emits PredictorChangedPayload via the EventBus,
    has no Qt signals. Predictor load and predict_freq are synchronous; they raise
    PredictorLoadError / PredictorNotLoaded for user-facing problems.
    """

    def __init__(
        self,
        state: SessionState,
        bus: BaseEventBus,
    ) -> None:
        self._state = state
        self._bus = bus
        self._predictor_path: str | None = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_predictor(self) -> FluxoniumPredictor | None:
        return self._state.exp_context.predictor

    def get_predictor_info(self) -> dict | None:
        """Return metadata about the current predictor, or None if not loaded.

        Keys: path, flux_bias, flux_half, flux_period, EJ, EC, EL.
        Exposing flux_half / flux_period lets the dialog build the affine
        conversions without importing FluxoniumPredictor; the EJ/EC/EL energies
        (predictor.params is the (EJ, EC, EL) GHz tuple) let the dialog read the
        active model back into its editable fields.
        """
        predictor = self._state.exp_context.predictor
        if predictor is None:
            return None
        ej, ec, el = predictor.params
        return {
            "path": self._predictor_path,
            "flux_bias": predictor.flux_bias,
            "flux_half": predictor.flux_half,
            "flux_period": predictor.flux_period,
            "EJ": ej,
            "EC": ec,
            "EL": el,
        }

    # ------------------------------------------------------------------
    # Predictor management (sync)
    # ------------------------------------------------------------------

    def load_predictor(self, req: LoadPredictorRequest) -> None:
        logger.info("load_predictor: path=%r flux_bias=%r", req.path, req.flux_bias)
        # path is Optional on the request (shared with the in-memory install path),
        # but the file-load path needs a concrete file — fail fast on None.
        if req.path is None:
            raise PredictorLoadError("load_predictor requires a file path, got None")
        try:
            predictor = FluxoniumPredictor.from_file(req.path, flux_bias=req.flux_bias)
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            raise PredictorLoadError(f"Failed to load predictor: {exc}") from exc
        self._predictor_path = req.path
        new_ctx = dataclasses.replace(self._state.exp_context, predictor=predictor)
        self._state.set_context(new_ctx)
        self._bus.emit(PredictorChangedPayload())

    def set_model_params(self, req: SetModelParamsRequest) -> None:
        """Build a FluxoniumPredictor from typed model params and install it.

        Routes through install_predictor (the in-memory seam) — NOT a file load —
        so a user/agent can plug trial energies directly. flux_period must be
        non-zero: the value<->flux affine divides by it (a zero period would make
        the mapping singular), so reject it fast with a clear message.
        """
        logger.info(
            "set_model_params: EJ=%r EC=%r EL=%r flux_half=%r flux_period=%r flux_bias=%r",
            req.EJ,
            req.EC,
            req.EL,
            req.flux_half,
            req.flux_period,
            req.flux_bias,
        )
        if req.flux_period == 0.0:
            raise PredictorLoadError(
                "flux_period must be non-zero (value<->flux affine)"
            )
        predictor = FluxoniumPredictor(
            (req.EJ, req.EC, req.EL),
            req.flux_half,
            req.flux_period,
            req.flux_bias,
        )
        self.install_predictor(predictor)

    def install_predictor(self, predictor: FluxoniumPredictor) -> None:
        """Install a ready-made predictor object (no file load).

        Used by the FLUX-AWARE-MOCK provisioner to install a predictor derived from
        the mock soc's SimParams. Shares the same write seam as ``load_predictor``
        (set_context + PredictorChangedPayload) so the View / MCP see it identically;
        the predictor path is cleared because this predictor has no backing file.
        """
        logger.info("install_predictor: %s", type(predictor).__name__)
        self._predictor_path = None
        new_ctx = dataclasses.replace(self._state.exp_context, predictor=predictor)
        self._state.set_context(new_ctx)
        self._bus.emit(PredictorChangedPayload())

    def clear_predictor(self) -> None:
        logger.info("clear_predictor")
        self._predictor_path = None
        new_ctx = dataclasses.replace(self._state.exp_context, predictor=None)
        self._state.set_context(new_ctx)
        self._bus.emit(PredictorChangedPayload())

    # ------------------------------------------------------------------
    # Prediction computation (sync, pure)
    # ------------------------------------------------------------------

    def predict_freq(self, req: PredictFreqRequest) -> float:
        predictor = self._state.exp_context.predictor
        if predictor is None:
            raise PredictorNotLoaded("No predictor loaded — load one first")
        return float(predictor.predict_freq(req.value, transition=req.transition))

    def predict_freq_curve(self, req: PredictCurveRequest) -> PredictCurveResult:
        """Compute f_ij vs device-value curves for multiple transitions in one pass.

        Uses calculate_energy_vs_flux (same backend as FluxoniumPredictor._predict_freq_array)
        so all transitions share a single diagonalisation sweep — O(n_values) eigen-
        solves rather than O(n_transitions * n_values). The value→flux conversion
        replicates the public scalar value_to_flux affine in vectorised form to
        avoid calling the private _value_to_flux_array.
        """
        predictor = self._state.exp_context.predictor
        if predictor is None:
            raise PredictorNotLoaded("No predictor loaded — load one first")
        if len(req.transitions) == 0:
            raise ValueError("transitions must not be empty")
        for frm, to in req.transitions:
            if frm < 0 or to < 0:
                raise ValueError(f"Transition levels must be >= 0, got ({frm}, {to})")
            if to < frm:
                raise ValueError(
                    f"Transition to-level must be >= from-level, got ({frm}, {to})"
                )

        # Vectorised value→flux (mirrors the public value_to_flux affine exactly).
        values = np.asarray(req.values, dtype=np.float64)
        fluxs = (
            values + predictor.flux_bias - predictor.flux_half
        ) / predictor.flux_period + 0.5

        from zcu_tools.simulate.fluxonium.energies import calculate_energy_vs_flux

        evals_count = max(to for _, to in req.transitions) + 5
        # cutoff=40 must match FluxoniumPredictor._predict_freq_array (predict.py:199)
        # to keep both code paths numerically consistent.
        _, energies = calculate_energy_vs_flux(
            predictor.params, fluxs, cutoff=40, evals_count=evals_count
        )
        # energies shape: (n_values, evals_count)

        freq_rows: list[NDArray[np.float64]] = []
        labels: list[str] = []
        for frm, to in req.transitions:
            diff = (energies[:, to] - energies[:, frm]) * 1e3  # GHz→MHz
            freq_rows.append(np.asarray(diff, dtype=np.float64))
            labels.append(f"{frm}→{to}")

        freqs_mhz = np.stack(freq_rows, axis=0)  # (n_transitions, n_values)
        return PredictCurveResult(
            labels=tuple(labels),
            values=values,
            fluxs=fluxs,
            freqs_mhz=freqs_mhz,
        )

    def predict_matrix_element_curve(
        self, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        """Compute |<i|op|j>| vs device-value curves for multiple transitions.

        Uses calculate_n_oper_vs_flux / calculate_phi_oper_vs_flux directly so
        there is no per-level cap — unlike FluxoniumPredictor.predict_matrix_element
        which hard-limits both levels to <=1.  return_dim is set to max(level)+1
        so higher transitions (e.g. 0->3) work correctly.
        """
        predictor = self._state.exp_context.predictor
        if predictor is None:
            raise PredictorNotLoaded("No predictor loaded — load one first")
        if len(req.transitions) == 0:
            raise ValueError("transitions must not be empty")
        for frm, to in req.transitions:
            if frm < 0 or to < 0:
                raise ValueError(f"Transition levels must be >= 0, got ({frm}, {to})")
            if to < frm:
                raise ValueError(
                    f"Transition to-level must be >= from-level, got ({frm}, {to})"
                )

        # Vectorised value→flux (mirrors predict_freq_curve affine exactly).
        values = np.asarray(req.values, dtype=np.float64)
        fluxs = (
            values + predictor.flux_bias - predictor.flux_half
        ) / predictor.flux_period + 0.5

        # Return dimension must cover the highest level in any transition.
        needed_dim = max(max(frm, to) for frm, to in req.transitions) + 1

        from zcu_tools.simulate.fluxonium.matrix_element import (
            calculate_n_oper_vs_flux,
            calculate_phi_oper_vs_flux,
        )

        calc = (
            calculate_n_oper_vs_flux
            if req.operator == "n"
            else calculate_phi_oper_vs_flux
        )
        _, opers = calc(predictor.params, fluxs, return_dim=needed_dim)
        # opers shape: (n_values, needed_dim, needed_dim), dtype complex

        mag_rows: list[NDArray[np.float64]] = []
        labels: list[str] = []
        for frm, to in req.transitions:
            mag = np.abs(opers[:, frm, to]).astype(np.float64)
            mag_rows.append(mag)
            labels.append(f"{frm}→{to}")

        mags = np.stack(mag_rows, axis=0)  # (n_transitions, n_values)
        return PredictMatrixCurveResult(
            labels=tuple(labels),
            values=values,
            fluxs=fluxs,
            mags=mags,
        )

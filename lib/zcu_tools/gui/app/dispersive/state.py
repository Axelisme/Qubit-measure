"""Passive state container for dispersive-fit-gui — shared by Controller and services.

Holds the single-flow analysis pipeline's state: the fluxonium inputs read from
a selected predictor file, the one loaded one-tone spectrum, its preprocessing result, and
the ``g`` / ``bare_rf`` tuning + fit state, plus the optimistic-concurrency
``VersionTable``. Like fluxdep-gui / measure-gui, every State write happens only
on the Qt main thread; workers never mutate State directly (their only side
effect is emitting a Qt signal whose main-thread slot writes here).

Unlike fluxdep (a *collection* of spectra), dispersive is single-onetone /
single-flow, so the aggregate holds one ``onetone`` rather than a dict. The
``fit_inputs`` field is the cross-app contract surface: it carries the
predictor file fluxdep-gui wrote (params + flux
alignment + a bare_rf seed), which dispersive consumes as its inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from zcu_tools.analysis.dispersive.models import PreprocessResult
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
from zcu_tools.gui.owner import OwnerThreadGuard

# VersionTable is the shared optimistic-concurrency mechanism (app-agnostic);
# re-exported so ``state.VersionTable`` stays resolvable. dispersive's key set
# (project / onetone / preprocess / fit) is documented beside the *_VERSION_KEY
# constants below.
from zcu_tools.gui.version_table import (
    VersionTable as VersionTable,  # (re-export)
)
from zcu_tools.resources.predictor_file import PredictorFlux

logger = logging.getLogger(__name__)

# Version-table resource keys (see VersionTable docstring for the bump↔drop map).
PROJECT_VERSION_KEY = "project"
ONETONE_VERSION_KEY = "onetone"
PREPROCESS_VERSION_KEY = "preprocess"
FIT_VERSION_KEY = "fit"

# Final bare_rf fallback when neither a dispersive file nor resonator supplies it.
DEFAULT_BARE_RF = 5.0


@dataclass(frozen=True)
class FluxoniumInputs:
    """Selected predictor parameters, dispersive's hard inputs.

    Produced by fluxdep-gui and read by dispersive (the cross-app handoff). The
    fluxonium ``params`` = (EJ, EC, EL) GHz and the recorded ``flux`` drive the
    dispersive simulation; ``bare_rf_seed`` is the initial resonator frequency
    (sourced from a prior dispersive section, else the fit's ``r_f``, else
    ``DEFAULT_BARE_RF``).
    """

    params: tuple[float, float, float]  # (EJ, EC, EL) GHz
    flux: PredictorFlux  # recorded offset in Phi0, signed periods and device units
    bare_rf_seed: float  # GHz


@dataclass
class OnetoneEntry:
    """One loaded one-tone spectrum (the single dispersive input spectrum).

    ``raw`` carries device values, projected Phi0, GHz frequencies and complex
    signals. ``device`` records the explicit source instrument or unnamed_device
    for unrecorded snapshots/Labber. ``unit`` retains the native source axis unit.
    Dispersive loads exactly one of these.
    """

    name: str
    raw: SpectrumData  # dev_values / fluxs / freqs(GHz) / signals(complex)
    device: str = "unnamed_device"  # source identity, not an inferred alias
    unit: str = "native"  # source instrument-axis unit, no automatic conversion


@dataclass
class DispFitState:
    """The manually-tuned ``g`` / ``bare_rf`` result + its simulation resolution.

    Frequencies are stored in **GHz** throughout (the slider UI converts to/from
    MHz for display). ``g`` is None until the user accepts a tuning ("Use these
    g/r_f"); ``has_result`` keys off it. ``res_dim`` is the resonator-truncation
    the simulation used (``qub_dim`` / ``qub_cutoff`` are fixed in the predictor);
    the prediction always covers the full flux axis (there is no down-sampling).
    """

    g: float | None = None  # GHz
    bare_rf: float | None = None  # GHz
    res_dim: int = 4

    @property
    def has_result(self) -> bool:
        return self.g is not None


class DispersiveState:
    """Passive GUI state container for the dispersive analysis pipeline."""

    def __init__(self, project: AnalysisTarget | None = None) -> None:
        self._owner_guard = OwnerThreadGuard()
        self.project = project
        self.predictor: str | None = None  # selected entry-local JSON filename
        self.fit_inputs: FluxoniumInputs | None = None
        self.onetone: OnetoneEntry | None = None
        self.preprocess: PreprocessResult | None = None
        self.disp_fit: DispFitState = DispFitState()
        self.version = VersionTable()

    # ------------------------------------------------------------------
    # Writers (services call these on the Qt main thread only).
    # ------------------------------------------------------------------

    def set_project(self, project: AnalysisTarget) -> None:
        """Select a target and clear inputs/results belonging to the previous one."""
        self._assert_owner()
        self.project = project
        self.predictor = None
        self.fit_inputs = None
        self.onetone = None
        self.preprocess = None
        self.disp_fit = DispFitState()
        self.version.drop_prefix(ONETONE_VERSION_KEY)
        self.version.drop_prefix(PREPROCESS_VERSION_KEY)
        self.version.drop_prefix(FIT_VERSION_KEY)
        self.version.bump(PROJECT_VERSION_KEY)

    def set_fit_inputs(
        self, inputs: FluxoniumInputs, *, predictor: str | None = None
    ) -> None:
        """Record predictor inputs and its local filename; seed the tuning bare_rf.

        Loading a predictor is a project-scoped change, so it bumps ``project``.
        The bare_rf seed only fills the tuning state when it has no value yet (a
        prior session's bare_rf is not clobbered by a re-read of the inputs).
        """
        self._assert_owner()
        self.fit_inputs = inputs
        self.predictor = predictor
        if self.disp_fit.bare_rf is None:
            self.disp_fit = replace(self.disp_fit, bare_rf=inputs.bare_rf_seed)
        self.version.bump(PROJECT_VERSION_KEY)
        logger.debug("set_fit_inputs: params=%s", inputs.params)

    def set_onetone(self, entry: OnetoneEntry) -> None:
        """Record the loaded one-tone; drops the stale preprocess + fit.

        New raw data invalidates any prior preprocessing (and the fit derived
        from it), so those version keys are dropped and the cached results
        cleared — a downstream reader never sees results for the old spectrum.
        """
        self._assert_owner()
        self.onetone = entry
        self.preprocess = None
        self.disp_fit = replace(self.disp_fit, g=None)
        self.version.bump(ONETONE_VERSION_KEY)
        self.version.drop_prefix(PREPROCESS_VERSION_KEY)
        self.version.drop_prefix(FIT_VERSION_KEY)
        logger.debug("set_onetone: name=%r", entry.name)

    def set_preprocess(self, result: PreprocessResult) -> None:
        """Record the preprocessing result; drop a fit from a different signature.

        Re-running preprocessing with different smoothing makes a prior fit stale,
        so a result whose ``signature`` differs from the new one invalidates the
        recorded fit.
        """
        self._assert_owner()
        prior_sig = self.preprocess.signature if self.preprocess is not None else None
        self.preprocess = result
        if prior_sig is not None and prior_sig != result.signature:
            self.disp_fit = replace(self.disp_fit, g=None)
            self.version.drop_prefix(FIT_VERSION_KEY)
        self.version.bump(PREPROCESS_VERSION_KEY)

    def set_disp_result(self, g: float, bare_rf: float, *, res_dim: int) -> None:
        """Record the manually-tuned g / bare_rf result + its simulation resolution."""
        self._assert_owner()
        self.disp_fit = replace(self.disp_fit, g=g, bare_rf=bare_rf, res_dim=res_dim)
        self.version.bump(FIT_VERSION_KEY)
        logger.debug("set_disp_result: g=%s bare_rf=%s", g, bare_rf)

    def _assert_owner(self) -> None:
        self._owner_guard.assert_owner()

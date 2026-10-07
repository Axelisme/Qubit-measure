"""Saved notebook run handles over the shared artifact persistence contract."""

from collections.abc import Mapping
from pathlib import Path
from typing import Generic, TypeVar

from matplotlib.figure import Figure

from zcu_tools.datafile import AnalysisDocument
from zcu_tools.experiment import (
    AcquisitionEvidence,
    AxesSpec,
    ExpCfgModel,
    GroupedAxesSpec,
    RunRecord,
)
from zcu_tools.experiment.run_saving import SaveReport
from zcu_tools.resources.entry import PointView, RoleView

CfgT = TypeVar("CfgT", bound=ExpCfgModel)
ResultT = TypeVar("ResultT")


class PartialSaveError(Exception, Generic[CfgT, ResultT]):
    """A latest-call output failure with its original SavedRun and report.

    run retains the captured context, run directory and content for an explicit
    subsequent save. report lists this call's successes/failures. There is no
    automatic retry. Ledger failures propagate separately as their original error.
    """

    def __init__(self, run: "SavedRun[CfgT, ResultT]", report: SaveReport) -> None:
        self.run = run
        self.report = report
        super().__init__(f"Some outputs failed for run {report.run_id}")


class SavedRun(Generic[CfgT, ResultT]):
    """Captured notebook content and evidence for explicitly repeated saves.

    Obtain this handle from NotebookExperiment.save_run. Its adapter verifies
    source identity before constructing it; there is no historical-source
    guessing. Saving does not recapture context, acquisition time or completion.
    """

    def __init__(
        self,
        ctx: PointView,
        *,
        source: RunRecord[CfgT, ResultT],
        spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
        acquisition: AcquisitionEvidence,
        roles: RoleView | None = None,
    ) -> None:
        """Capture adapter-validated source/spec/evidence and this point's context.

        Adapter-internal construction: source is its current successful record
        and acquisition is acquisition_for(source). spec is the core's declared
        persistence mapping. roles, if given, is resolved from ctx; None records
        no roles. Snapshot only cached context and prepare typed data. Do not
        create run folders or append events until save. Invalid source/spec
        raises TypeError/ValueError; bound storage errors propagate.
        """
        raise NotImplementedError

    @property
    def run_id(self) -> str:
        """Return the acquisition's immutable run identity."""
        raise NotImplementedError

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return successful destinations from the latest report only."""
        raise NotImplementedError

    @property
    def report(self) -> SaveReport:
        """Return this handle's latest output report; no ledger receipt is stored."""
        raise NotImplementedError

    def save(
        self,
        *,
        figures: Mapping[str, Figure] | None = None,
        analysis: tuple[AnalysisDocument, ...] | None = None,
        comment: str | None = None,
    ) -> SaveReport:
        """Rewrite data and retained/supplied files under the same run identity.

        Named figures become run PNGs plus point copies. Each analysis document
        becomes analysis/<pane>.json. Pass typed content to the shared saver;
        encoding failures affect that member only, not data or other files.
        None preserves retained content/comment;
        empty mapping/tuple clears that set. comment is raw user text.
        Append acquired once before the first save and saved once per call.
        PartialSaveError retains this same handle and the latest report.
        Ledger errors propagate directly; do not retry, roll back or recapture.
        """
        raise NotImplementedError

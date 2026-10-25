"""Shared execution, native saving and partial-result handling for lab stages."""

from dataclasses import dataclass
from typing import Literal

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.axes_spec import AxesSpec, GroupedAxesSpec
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.operations import RunFunction
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.plotting.plots import Plots

from zcu_lab.workflows.common.saving import save_native
from zcu_lab.workflows.common.summary import SummaryAccumulator, observe_operation


@dataclass(frozen=True)
class StageEnv:
    """Run-level services shared by stages, not their scientific inputs.

    plots holds workflow figures. snapshot is the captured entry/point identity
    for native saving. summary collects delivered operation facts, or None when
    no run report is requested. Construct once per workflow run; stages borrow
    these services without changing the point record or accepted histories.
    """

    plots: Plots
    snapshot: RunSnapshot
    summary: SummaryAccumulator | None = None


@dataclass(frozen=True)
class SavedAcquisition[Result]:
    """Acquisition outcome after saving every Completed, including partials.

    status is completed, partial or failed. result is available only on complete
    acquisition, so partial data never enters a fit. file is the relative native
    filename after saving, or None on acquisition failure. reason is stopped for
    partial, the acquisition reason for failed, and None for completed.
    """

    status: Literal["completed", "partial", "failed"]
    result: Result | None = None
    file: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class NativeOutput[Cfg: ConfigBase, Result]:
    """Native destination: relative filename and the core's serialization spec."""

    filename: str
    spec: AxesSpec[Result, Cfg] | GroupedAxesSpec[Result, Cfg]


async def acquire_saved[Cfg: ConfigBase, Result](
    scope: Scope,
    env: StageEnv,
    experiment: RunFunction[Cfg, Result],
    cfg: Cfg,
    *,
    name: str,
    output: NativeOutput[Cfg, Result],
) -> SavedAcquisition[Result]:
    """Run a core experiment, observe facts and save raw before exposing data.

    cfg is the detached native configuration; name labels the operation.
    output pairs a filename relative to scope.dir with its native axes mapping.
    Failed acquisitions have no file. Partial completions save raw then return
    status=partial without fit data. Complete results are returned unchanged.
    Acquisition, writer and unexpected framework exceptions propagate.
    """
    outcome = await scope.run(experiment, cfg, name=name)
    observe_operation(env.summary, outcome)
    if isinstance(outcome, Failed):
        return SavedAcquisition(status="failed", reason=outcome.reason)
    save_native(
        outcome,
        scope.dir / output.filename,
        spec=output.spec,
        snapshot=env.snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return SavedAcquisition(
            status="partial", file=output.filename, reason="stopped"
        )
    return SavedAcquisition(
        status="completed", result=outcome.value.result, file=output.filename
    )

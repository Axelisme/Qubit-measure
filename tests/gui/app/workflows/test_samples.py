"""Grouped captured export identity checks occur before native replacement."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from zcu_tools.experiment.workflows import CommittedRecord, EnginePorts, RunPaths
from zcu_tools.experiment.workflows.exporting import (
    FluxdepExportSnapshot,
    PreparedFluxdepExport,
)
from zcu_tools.gui.app.workflows.exporting import (
    FluxdepExportBinding,
    WorkflowExportBindings,
)
from zcu_tools.gui.app.workflows.samples import SampleExportBinding
from zcu_tools.resources.entry import ResultEntry

from tests._native_support import native_metadata
from tests.experiment.workflows._engine_fakes import Plan, Rig, Tunables

from ._controller_support import create_binding, create_controller, declare, identity
from ._start_fakes import StartHarness


def unusable_science(*, records: Sequence[CommittedRecord]) -> PreparedFluxdepExport:
    raise AssertionError("Mismatched Start must never call science")


@pytest.mark.parametrize("kind", ["fluxdep", "samples"])
def test_wrong_export_identity_never_replaces_terminal_run(
    tmp_path: Path, kind: str
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)
    ports = EnginePorts(
        rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
    )
    controller.start(
        declare(rig),
        plan=Plan(),
        tunables=Tunables(),
        ports=ports,
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.complete_segment()
    previous = controller.status()
    paths = RunPaths(tmp_path / "next-metadata", tmp_path / "next-data")
    entry = ResultEntry.create(
        "captured",
        result_root=tmp_path / "results",
        database_root=tmp_path / "database",
    )
    if kind == "fluxdep":
        exports = WorkflowExportBindings(
            fluxdep=FluxdepExportBinding(
                FluxdepExportSnapshot(
                    identity("wrong"),
                    paths,
                    native_metadata().snapshot,
                    rig.clock.now(),
                    None,
                ),
                unusable_science,
                entry.result_path,
                entry.database_path,
                entry.ledger,
            )
        )
    else:
        exports = WorkflowExportBindings(
            samples=SampleExportBinding(
                "wrong",
                "captured",
                entry.samples("Q1"),
                None,
                "No sample science",
            )
        )
    with pytest.raises(ValueError, match="identity"):
        controller.start(
            declare(rig),
            plan=Plan(),
            tunables=Tunables(),
            ports=ports,
            binding=create_binding("next"),
            paths=paths,
            identity=identity("next"),
            export_bindings=exports,
        )
    assert controller.status() == previous and controller.run_id == "run"
    assert not paths.metadata_root.exists() and not paths.data_root.exists()
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0

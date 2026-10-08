"""Native Save RPC reports backed by real writers, ledger and State."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.remote.handlers.run_save import (
    h_tab_save_artifacts,
    h_tab_save_data,
)
from zcu_tools.gui.app.measure.remote.save_types import SaveSnapshotWire
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.expected_error import FailedPreconditionError

from tests.gui.app.measure._native_save_support import (
    declare_panes,
    native_save_session,
)


@pytest.mark.parametrize("selection", ["data", "all", "analysis", "post_analysis"])
def test_save_rpc_returns_the_current_complete_report(
    tmp_path: Path, qapp: QApplication, selection: str
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        declare_panes(session)
        adapter = MagicMock()
        adapter.render_view = None
        adapter.save_control = session.control
        adapter.tab_control.get_tab_snapshot.side_effect = TabService(
            session.state,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ).get_snapshot
        if selection == "data":
            reply = h_tab_save_data(
                adapter, {"tab_id": "tab", "comment": None, "run_operation_id": None}
            )
            expected = {"run": {"measurement_data": 2}}
        else:
            refs = (
                "all"
                if selection == "all"
                else [{"section": selection, "name": "results"}]
            )
            reply = h_tab_save_artifacts(
                adapter,
                {
                    "tab_id": "tab",
                    "comment": None,
                    "run_operation_id": None,
                    "operation_id": None,
                    "artifacts": refs,
                },
            )
            expected = (
                {
                    "run": {"measurement_data": 2},
                    "analysis": {"results": 5},
                    "post_analysis": {"results": 1},
                }
                if selection == "all"
                else {selection: {"results": 5 if selection == "analysis" else 1}}
            )
        report = SaveSnapshotWire.model_validate(reply)
        assert report.run_id == session.metadata.run_id
        assert {
            section: {
                name: sum(len(outputs) for outputs in artifact.members.values())
                for name, artifact in artifacts.items()
            }
            for section, artifacts in report.artifacts.items()
        } == expected
        for artifacts in report.artifacts.values():
            for artifact in artifacts.values():
                assert artifact.status == "Saved"
                for outputs in artifact.members.values():
                    for output in outputs:
                        assert output.status == "saved"
                        assert Path(output.path).is_file()
        assert len(session.entry.ledger.events()) == 1


@pytest.mark.parametrize("block_native", [True, False])
def test_save_rpc_reports_partial_and_initial_all_failed_identity(
    tmp_path: Path, qapp: QApplication, block_native: bool
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        (session.entry.database_path / "Labber").write_text("blocked")
        if block_native:
            (
                session.entry.database_path
                / "runs"
                / session.metadata.run_id
                / "data.h5"
            ).mkdir(parents=True)
        adapter = MagicMock()
        adapter.save_control = session.control
        adapter.tab_control.get_tab_snapshot.side_effect = TabService(
            session.state,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ).get_snapshot
        report = SaveSnapshotWire.model_validate(
            h_tab_save_data(
                adapter, {"tab_id": "tab", "comment": None, "run_operation_id": None}
            )
        )
        assert report.run_id == (None if block_native else session.metadata.run_id)
        artifact = report.artifacts["run"]["measurement_data"]
        outputs = artifact.members["data"]
        assert [output.status for output in outputs] == (
            ["failed", "failed"] if block_native else ["saved", "failed"]
        )
        assert artifact.status == ("Not saved" if block_native else "Partially saved")
        assert all(output.error for output in outputs if output.status == "failed")


def test_data_save_rpc_rejects_a_replaced_run_before_writes(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        adapter = MagicMock()
        adapter.save_control = session.control
        with pytest.raises(FailedPreconditionError, match="no longer contains"):
            h_tab_save_data(
                adapter, {"tab_id": "tab", "comment": None, "run_operation_id": 42}
            )
        assert session.entry.ledger.events() == ()
        assert not (session.entry.database_path / "runs").exists()

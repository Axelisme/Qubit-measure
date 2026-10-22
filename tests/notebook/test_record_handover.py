"""Notebook save hands one record and its fixed evidence to artifact persistence."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.artifacts import SaveReport
from zcu_tools.experiment.records import RunRecord
from zcu_tools.notebook import NotebookAdapter, saving
from zcu_tools.plotting.plots import NonPresentingHost
from zcu_tools.resources.entry import ResultEntry

from tests.experiment._saving_support import SavingCfg, SavingCore


def test_save_merges_acquisition_and_point_metadata_into_the_handed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    point = ResultEntry.create(
        "record-entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
    ).new_point("sweet", description="captured point")
    exp = NotebookAdapter(
        soc=object(),
        soccfg=object(),
        device_manager=DeviceManager(),
        host=NonPresentingHost(),
    )(SavingCore())
    source = exp.run(SavingCfg())
    evidence = exp.acquisition_for(source)
    saver = MagicMock(return_value=SaveReport(evidence.run_id, ()))
    monkeypatch.setattr(saving, "save_artifacts", saver)
    exp.save_run(point, source=source)
    call = saver.call_args
    assert call is not None
    record = call.kwargs.get("record")
    assert isinstance(record, RunRecord), "Artifact persistence must receive the record"
    assert record.metadata is not None
    assert record.result is source.result
    assert record.metadata.run_id == evidence.run_id
    assert record.metadata.started_at == evidence.started_at.isoformat()
    assert record.metadata.finished_at == evidence.finished_at.isoformat()
    assert record.metadata.provenance == evidence.provenance
    assert record.metadata.snapshot.entry_name == "record-entry"
    assert record.metadata.snapshot.point == "sweet"
    assert record.metadata.device_snapshot == {}

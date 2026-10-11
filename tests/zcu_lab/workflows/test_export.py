"""Terminal fluxdep export uses committed Engine evidence and public readers."""

from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from zcu_tools.datafile import DataVariable, load_grouped_labber_data, load_run_data
from zcu_tools.experiment import workflows as framework
from zcu_tools.gui.app import workflows as host
from zcu_tools.resources.entry import RecordsLedger, SavedPayload, SaveLayout

from zcu_lab import workflows as science

from ._export_inputs import RUN_ID, SAVED_AT, synthetic_run
from ._sim_inputs import Inputs


def test_t1_adapter_keeps_actual_delay_rows_and_rejected_fit(
    inputs: Inputs, tmp_path: Path
) -> None:
    engine, plan, _identity, paths = synthetic_run(inputs, tmp_path, "stopped")
    assert engine.execute(RUN_ID).lifecycle == "stopped"
    records = engine.records(RUN_ID)
    assert len(records) == 3
    prepared = science.prepare_t1_fluxdep_export(
        plan=plan, data_root=paths.data_root, records=records
    )
    variables = prepared.payload.variables
    assert tuple(variables) == ("signal", "time", "t1")
    np.testing.assert_array_equal(
        variables["signal"].z,
        [
            [1 + 2j, 2 + 3j, complex(np.nan, np.nan)],
            [4 + 5j, 6 + 7j, complex(np.nan, np.nan)],
            [complex(np.nan, np.nan)] * 3,
            [complex(np.nan, np.nan)] * 3,
        ],
    )
    np.testing.assert_array_equal(
        variables["time"].z,
        [[0, 1e-6, 2e-6], [0, 3e-6, np.nan], [np.nan] * 3, [np.nan] * 3],
    )
    np.testing.assert_array_equal(variables["t1"].z, [20e-6, np.nan, np.nan, np.nan])
    assert [(a.name, a.unit) for a in variables["signal"].axes] == [
        ("Time Index", ""),
        ("Flux", "native"),
    ]
    np.testing.assert_array_equal(variables["signal"].axes[0].values, [0, 1, 2])
    np.testing.assert_array_equal(
        variables["signal"].axes[1].values, [1, 1.1, 1.2, 1.3]
    )
    assert [(a.name, a.unit) for a in variables["t1"].axes] == [("Flux", "native")]
    assert prepared.cfg.values == plan.model_dump(mode="json")
    assert (prepared.cfg.cfg_type, prepared.cfg.schema_version) == ("T1Plan", "1.0")
    with pytest.raises(ValueError, match="usable committed"):
        science.prepare_t1_fluxdep_export(
            plan=plan, data_root=paths.data_root, records=[]
        )


@pytest.mark.parametrize("terminal", ["done", "stopped", "failed", "aborted"])
def test_terminal_export_roundtrip_overwrite_new_labber_and_ledger(
    inputs: Inputs,
    tmp_path: Path,
    terminal: Literal["done", "stopped", "failed", "aborted"],
) -> None:
    engine, plan, identity, paths = synthetic_run(inputs, tmp_path, terminal)
    assert engine.execute(RUN_ID).lifecycle == terminal
    snapshot = framework.FluxdepExportSnapshot(
        identity, paths, plan.snapshot, SAVED_AT, SAVED_AT
    )
    record_root = tmp_path / "records"
    record_root.mkdir()
    ledger = RecordsLedger(record_root, entry_id=str(plan.snapshot.entry_id))
    prepare = partial(
        science.prepare_t1_fluxdep_export, plan=plan, data_root=paths.data_root
    )
    canonical = engine.records(RUN_ID)
    reports = []
    for _ in range(2):
        layout = SaveLayout(
            result_path=tmp_path / "result",
            database_path=tmp_path / "database",
            run_id=RUN_ID,
            point=plan.snapshot.point,
            saved_at=SAVED_AT,
        )
        report = host.export_fluxdep_run(
            RUN_ID,
            engine=engine,
            snapshot=snapshot,
            prepare=prepare,
            layout=layout,
            ledger=ledger,
            origin="gui",
            saved_at=SAVED_AT,
        )
        reports.append(report)
        assert report.ledger_error is None
        assert [item.status for item in report.outputs] == ["saved", "saved"]
        native = load_run_data(paths.data_root / "merged.h5")
        labber = load_grouped_labber_data(
            str(report.outputs[1].path),
            required_variables=("signal", "time", "t1"),
        )
        for name in map(DataVariable, ("signal", "time", "t1")):
            a, b = native.payload.variables[name], labber.variables[name]
            np.testing.assert_array_equal(a.z, b.z)
            assert [(axis.name, axis.unit) for axis in a.axes] == [
                (axis.name, axis.unit) for axis in b.axes
            ]
            assert (a.data.name, a.data.unit) == (b.data.name, b.data.unit)
            for native_axis, labber_axis in zip(a.axes, b.axes, strict=True):
                np.testing.assert_array_equal(native_axis.values, labber_axis.values)
        assert native.metadata.run_id == RUN_ID
        assert native.metadata.provenance == identity.provenance
        assert native.metadata.snapshot == plan.snapshot
        assert native.metadata.experiment == "T1_fluxdep"
        assert native.cfg.values == plan.model_dump(mode="json")
        assert native.metadata.device_snapshot is None
        np.testing.assert_array_equal(
            native.payload.variables["t1"].z, [20e-6, np.nan, np.nan, np.nan]
        )
        assert native.payload.variables["signal"].z.shape == (4, 3)
    assert (
        reports[0].outputs[0].path
        == reports[1].outputs[0].path
        == paths.data_root / "merged.h5"
    )
    assert reports[0].outputs[1].path != reports[1].outputs[1].path
    assert "Labber/2026/10/Data_1006" in str(reports[0].outputs[1].path)
    events = ledger.events()
    assert len(events) == 2
    for event, report in zip(events, reports, strict=True):
        assert event.kind == "saved" and event.origin == "gui"
        assert isinstance(event.payload, SavedPayload)
        assert event.payload.labber_path == str(report.outputs[1].path)
        assert [out.path for out in event.payload.outputs] == [
            str(out.path) for out in report.outputs
        ]
    assert engine.records(RUN_ID) == canonical
    assert engine.status() is not None and engine.status().lifecycle == terminal


@pytest.mark.parametrize("failure", ["native", "labber", "ledger"])
def test_output_and_ledger_failures_retain_independent_successes(
    inputs: Inputs, tmp_path: Path, failure: str
) -> None:
    engine, plan, identity, paths = synthetic_run(inputs, tmp_path)
    engine.execute(RUN_ID)
    snapshot = framework.FluxdepExportSnapshot(
        identity, paths, plan.snapshot, SAVED_AT, None
    )
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    ledger = RecordsLedger(records_dir, entry_id=str(plan.snapshot.entry_id))
    database = tmp_path / "database"
    if failure == "native":
        (paths.data_root / "merged.h5").mkdir()
    elif failure == "labber":
        database.mkdir()
        (database / "Labber").write_text("blocked")
    else:
        (records_dir / "ledger.jsonl").mkdir()
    layout = SaveLayout(
        result_path=tmp_path / "result",
        database_path=database,
        run_id=RUN_ID,
        point=plan.snapshot.point,
        saved_at=SAVED_AT,
    )
    report = host.export_fluxdep_run(
        RUN_ID,
        engine=engine,
        snapshot=snapshot,
        prepare=partial(
            science.prepare_t1_fluxdep_export, plan=plan, data_root=paths.data_root
        ),
        layout=layout,
        ledger=ledger,
        origin="mcp",
        saved_at=SAVED_AT,
    )
    assert [item.status for item in report.outputs] == {
        "native": ["failed", "saved"],
        "labber": ["saved", "failed"],
        "ledger": ["saved", "saved"],
    }[failure]
    assert bool(report.ledger_error) == (failure == "ledger")
    for out in report.outputs:
        if out.status == "saved":
            assert out.error is None
            if out.format == "data_h5":
                assert load_run_data(out.path).metadata.finished_at is None
            else:
                assert load_grouped_labber_data(str(out.path)).variables[
                    "signal"
                ].z.shape == (4, 3)
        else:
            assert out.error
    assert engine.status() is not None and engine.status().lifecycle == "done"


@pytest.mark.parametrize("state", ["running", "paused", "mismatch", "unsupported"])
def test_export_admission_rejects_before_writes(
    inputs: Inputs, tmp_path: Path, state: str
) -> None:
    engine, plan, identity, paths = synthetic_run(inputs, tmp_path)
    if state == "paused":
        engine.pause(RUN_ID)
        engine.execute(RUN_ID)
    elif state in {"mismatch", "unsupported"}:
        engine.execute(RUN_ID)
    snapshot = framework.FluxdepExportSnapshot(
        identity, paths, plan.snapshot, SAVED_AT, None
    )
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    ledger = RecordsLedger(records_dir, entry_id=str(plan.snapshot.entry_id))
    layout = SaveLayout(
        result_path=tmp_path / "result",
        database_path=tmp_path / "database",
        run_id=RUN_ID,
        point=plan.snapshot.point,
        saved_at=SAVED_AT,
    )
    with pytest.raises(
        framework.RunMismatch
        if state == "mismatch"
        else framework.InvalidRunState
        if state in {"running", "paused"}
        else ValueError
    ):
        host.export_fluxdep_run(
            "stale" if state == "mismatch" else RUN_ID,
            engine=engine,
            snapshot=snapshot,
            prepare=None
            if state == "unsupported"
            else partial(
                science.prepare_t1_fluxdep_export, plan=plan, data_root=paths.data_root
            ),
            layout=layout,
            ledger=ledger,
            origin="gui",
            saved_at=SAVED_AT,
        )
    assert not (paths.data_root / "merged.h5").exists()
    assert not (tmp_path / "database").exists()
    assert ledger.events() == ()


def test_labber_allocation_failure_keeps_native_and_full_ledger_record(
    inputs: Inputs, tmp_path: Path
) -> None:
    engine, plan, identity, paths = synthetic_run(inputs, tmp_path)
    engine.execute(RUN_ID)
    snapshot = framework.FluxdepExportSnapshot(
        identity, paths, plan.snapshot, SAVED_AT, None
    )
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    ledger = RecordsLedger(records_dir, entry_id=str(plan.snapshot.entry_id))
    database = tmp_path / "database"
    database.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (database / "Labber").symlink_to(outside, target_is_directory=True)
    layout = SaveLayout(
        result_path=tmp_path / "result",
        database_path=database,
        run_id=RUN_ID,
        point=plan.snapshot.point,
        saved_at=SAVED_AT,
    )
    report = host.export_fluxdep_run(
        RUN_ID,
        engine=engine,
        snapshot=snapshot,
        prepare=partial(
            science.prepare_t1_fluxdep_export, plan=plan, data_root=paths.data_root
        ),
        layout=layout,
        ledger=ledger,
        origin="gui",
        saved_at=SAVED_AT,
    )
    assert report.outputs[0].status == "saved"
    assert report.outputs[0].path == paths.data_root / "merged.h5"
    assert load_run_data(report.outputs[0].path).payload.variables[
        DataVariable("signal")
    ].z.shape == (4, 3)
    assert report.outputs[1].path is None
    assert report.outputs[1].status == "failed" and report.outputs[1].error
    assert report.ledger_error is None
    event = ledger.events()[0]
    assert isinstance(event.payload, SavedPayload)
    assert len(event.payload.outputs) == 1
    assert event.payload.outputs[0].path == str(report.outputs[0].path)
    entry = ledger.get(event.id)
    assert entry.record is not None
    assert entry.record["outputs"][1]["path"] is None
    assert entry.record["outputs"][1]["error"] == report.outputs[1].error
    assert list(outside.iterdir()) == []

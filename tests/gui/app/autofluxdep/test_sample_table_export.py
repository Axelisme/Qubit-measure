"""SampleTable v2 export tests for autofluxdep run artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QCoreApplication, QThread  # type: ignore[attr-defined]
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.experiments._support.result import QubitFreqResult
from zcu_tools.gui.app.autofluxdep.experiments.t1 import T1Builder
from zcu_tools.gui.app.autofluxdep.experiments.t2echo import T2EchoBuilder
from zcu_tools.gui.app.autofluxdep.experiments.t2ramsey import T2RamseyBuilder
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.orchestrator import InfoStore
from zcu_tools.gui.app.autofluxdep.services import (
    sample_table_export as sample_table_export_module,
)
from zcu_tools.gui.app.autofluxdep.services.run_store import RunStore
from zcu_tools.gui.app.autofluxdep.services.sample_table_export import (
    export_sample_table_from_artifact,
    sample_rows_from_journal,
)
from zcu_tools.gui.app.autofluxdep.state import ProjectInfo

from ._helpers import make_builder, place

_SAMPLE_DATE = "2026-07-06 12:34:56"

_EXPECTED_COLUMNS = [
    "dev_value",
    "dev_unit",
    "Freq (MHz)",
    "T1 (us)",
    "T1err (us)",
    "T2r (us)",
    "T2r err (us)",
    "T2e (us)",
    "T2e err (us)",
    "date",
]


@pytest.fixture(autouse=True)
def _fixed_sample_date(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object | None = None) -> datetime:
            del tz
            return cls(2026, 7, 6, 12, 34, 56)

    monkeypatch.setattr(sample_table_export_module, "datetime", FixedDateTime)


def _project(tmp_path: Path) -> ProjectInfo:
    return ProjectInfo(
        chip_name="chip",
        qub_name="q1",
        result_dir=str(tmp_path / "results"),
        database_path=str(tmp_path / "Database" / "chip" / "q1"),
        params_path=str(tmp_path / "params.json"),
    )


def _node_names() -> tuple[str, ...]:
    return ("qubit_freq", "lenrabi", "t1", "t2ramsey", "t2echo")


def _store(
    tmp_path: Path,
    *,
    results: dict[str, object] | None = None,
    flux_device_name: str | None = "fake_flux",
    flux_unit: str = "A",
) -> RunStore:
    nodes = [place(make_builder(name)) for name in _node_names()]
    return RunStore.create(
        project=_project(tmp_path),
        flux_values=[0.0, 0.5],
        flux_device_name=flux_device_name,
        flux_unit=flux_unit,
        nodes=nodes,
        results=results or {},
    )


def _commit_full_first_point(store: RunStore) -> None:
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.write_node_row(
        "lenrabi",
        0,
        Patch({"pi_length": 0.12, "pi2_length": 0.06, "rabi_len": 0.12}),
        InfoStore(),
    )
    store.write_node_row("t1", 0, Patch({"t1": 12.0, "t1err": 0.4}), InfoStore())
    store.write_node_row(
        "t2ramsey", 0, Patch({"t2r": 21.0, "t2r_err": 0.5}), InfoStore()
    )
    store.write_node_row("t2echo", 0, Patch({"t2e": 31.0, "t2e_err": 0.6}), InfoStore())
    store.commit_flux(0, 0.125, InfoStore())


def test_export_sample_table_uses_notebook_keys_and_committed_rows(tmp_path):
    store = _store(tmp_path)
    _commit_full_first_point(store)
    store.write_node_row("qubit_freq", 1, Patch({"qubit_freq": 6001.0}), InfoStore())
    store.finalize("finished")

    result = export_sample_table_from_artifact(store.run_dir)

    assert result.row_count == 1
    df = pd.read_csv(result.path)
    assert list(df.columns) == _EXPECTED_COLUMNS
    assert len(df) == 1
    row = df.iloc[0]
    assert row["dev_value"] == 0.125
    assert row["dev_unit"] == "A"
    assert row["Freq (MHz)"] == 5001.25
    assert row["T1 (us)"] == 12.0
    assert row["T1err (us)"] == 0.4
    assert row["T2r (us)"] == 21.0
    assert row["T2r err (us)"] == 0.5
    assert row["T2e (us)"] == 31.0
    assert row["T2e err (us)"] == 0.6
    assert row["date"] == _SAMPLE_DATE
    assert {"flux", "flux_int", "flux_period"}.isdisjoint(df.columns)
    assert "comment" not in df.columns
    assert "pi_length" not in df.columns
    assert "rabi_len" not in df.columns


def test_export_sample_table_appends_by_default(tmp_path):
    store = _store(tmp_path)
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.write_node_row("t1", 0, Patch({"t1": 12.0}), InfoStore())
    store.commit_flux(0, 0.0, InfoStore())
    store.finalize("stopped")

    output = tmp_path / "custom_samples.csv"
    pd.DataFrame(
        [
            {
                "dev_value": -1.0,
                "dev_unit": "A",
                "Freq (MHz)": 4000.0,
                "date": "2026-07-01 00:00:00",
            }
        ]
    ).to_csv(output, index=False)
    result = export_sample_table_from_artifact(store.manifest_path, filepath=output)

    assert result.path == str(output)
    assert result.row_count == 1
    df = pd.read_csv(output)
    assert len(df) == 2
    assert df.loc[0, "dev_value"] == -1.0
    assert df.loc[0, "dev_unit"] == "A"
    assert df.loc[0, "Freq (MHz)"] == 4000.0
    assert df.loc[0, "date"] == "2026-07-01 00:00:00"
    assert df.loc[1, "dev_value"] == 0.0
    assert df.loc[1, "dev_unit"] == "A"
    assert df.loc[1, "Freq (MHz)"] == 5001.25
    assert df.loc[1, "T1 (us)"] == 12.0
    assert df.loc[1, "date"] == _SAMPLE_DATE
    assert "comment" not in df.columns


def test_export_sample_table_can_overwrite_existing_file(tmp_path):
    store = _store(tmp_path)
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.commit_flux(0, 0.0, InfoStore())
    store.finalize("stopped")

    output = tmp_path / "custom_samples.csv"
    pd.DataFrame([{"calibrated mA": -1.0, "Freq (MHz)": 4000.0}]).to_csv(
        output, index=False
    )
    result = export_sample_table_from_artifact(
        store.manifest_path,
        filepath=output,
        append=False,
    )

    assert result.path == str(output)
    df = pd.read_csv(output)
    assert df.to_dict(orient="records") == [
        {
            "dev_value": 0.0,
            "dev_unit": "A",
            "Freq (MHz)": 5001.25,
            "date": _SAMPLE_DATE,
        }
    ]


def test_export_sample_table_accepts_paired_data_run_directory(tmp_path):
    store = _store(tmp_path)
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.commit_flux(0, 0.0, InfoStore())
    store.finalize("finished")

    result = export_sample_table_from_artifact(store.data_dir)

    df = pd.read_csv(result.path)
    assert df.to_dict(orient="records") == [
        {
            "dev_value": 0.0,
            "dev_unit": "A",
            "Freq (MHz)": 5001.25,
            "date": _SAMPLE_DATE,
        }
    ]


def test_decay_nodes_declare_sample_error_patch_keys():
    assert T1Builder().provides == ("t1", "t1err")
    assert T2RamseyBuilder().provides == ("t2r", "t2r_err", "t2r_detune")
    assert T2EchoBuilder().provides == ("t2e", "t2e_err")


def test_export_sample_table_falls_back_to_qubit_freq_row_summary(tmp_path):
    flux = np.array([0.0, 0.5], dtype=float)
    result = QubitFreqResult.allocate(flux, np.array([-1.0, 0.0, 1.0], dtype=float))
    result.predict_freq[0] = 5120.0
    result.fit_freq[0] = 5123.0
    store = _store(tmp_path, results={"qubit_freq": result})
    store.write_node_row("qubit_freq", 0, Patch(), InfoStore())
    store.commit_flux(0, 0.0, InfoStore())
    store.finalize("finished")

    exported = export_sample_table_from_artifact(store.run_dir)

    df = pd.read_csv(exported.path)
    assert df.to_dict(orient="records") == [
        {
            "dev_value": 0.0,
            "dev_unit": "A",
            "Freq (MHz)": 5123.0,
            "date": _SAMPLE_DATE,
        }
    ]


def test_export_sample_table_uses_snapshot_unit_for_voltage_device(tmp_path):
    store = _store(tmp_path, flux_unit="V")
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.commit_flux(0, 1.25, InfoStore())
    store.finalize("finished")

    exported = export_sample_table_from_artifact(store.run_dir)

    df = pd.read_csv(exported.path)
    assert df.to_dict(orient="records") == [
        {
            "dev_value": 1.25,
            "dev_unit": "V",
            "Freq (MHz)": 5001.25,
            "date": _SAMPLE_DATE,
        }
    ]


def test_sample_rows_from_journal_requires_authoritative_unit():
    with pytest.raises(ValueError, match="authoritative"):
        sample_rows_from_journal([], dev_unit="", date=_SAMPLE_DATE)
    with pytest.raises(ValueError, match="authoritative"):
        sample_rows_from_journal([], dev_unit="mA", date=_SAMPLE_DATE)


def test_export_sample_table_rejects_non_terminal_run(tmp_path):
    store = _store(tmp_path)
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.commit_flux(0, 0.0, InfoStore())

    with pytest.raises(ValueError, match="non-terminal"):
        export_sample_table_from_artifact(store.run_dir)


def test_export_sample_table_rejects_runs_without_completed_sample_rows(tmp_path):
    store = _store(tmp_path)
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.finalize("finished")

    with pytest.raises(ValueError, match="no completed flux points"):
        export_sample_table_from_artifact(store.run_dir)


def test_export_sample_table_rejects_legacy_empty_unit_artifact_before_mutation(
    tmp_path,
):
    store = _store(tmp_path, flux_unit="")
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.commit_flux(0, 0.0, InfoStore())
    store.finalize("finished")
    output = tmp_path / "samples.csv"
    output.write_text("keep-me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="authoritative"):
        export_sample_table_from_artifact(store.manifest_path, filepath=output)

    assert output.read_text(encoding="utf-8") == "keep-me\n"


def test_export_sample_table_bare_sweep_artifact_fails_before_creating_csv(
    tmp_path,
):
    store = _store(tmp_path, flux_device_name=None, flux_unit="")
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.commit_flux(0, 0.0, InfoStore())
    store.finalize("finished")

    with pytest.raises(ValueError, match="authoritative"):
        export_sample_table_from_artifact(store.run_dir)

    assert not (store.data_dir / "exports" / "sample" / "samples.csv").exists()


def test_export_sample_table_rejects_unsupported_device_artifact_before_mutation(
    tmp_path,
):
    store = _store(tmp_path, flux_device_name="yoko", flux_unit="")
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.commit_flux(0, 0.0, InfoStore())
    store.finalize("finished")
    output = tmp_path / "samples.csv"
    output.write_text("keep-me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="authoritative"):
        export_sample_table_from_artifact(store.manifest_path, filepath=output)

    assert output.read_text(encoding="utf-8") == "keep-me\n"


def test_export_sample_table_rejects_legacy_existing_table_before_mutation(tmp_path):
    store = _store(tmp_path)
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.commit_flux(0, 0.0, InfoStore())
    store.finalize("finished")
    output = tmp_path / "legacy_samples.csv"
    pd.DataFrame([{"calibrated mA": -1.0, "Freq (MHz)": 4000.0}]).to_csv(
        output, index=False
    )
    before = output.read_bytes()

    with pytest.raises(ValueError, match="stale legacy"):
        export_sample_table_from_artifact(store.manifest_path, filepath=output)

    assert output.read_bytes() == before


def test_export_sample_table_rejects_invalid_existing_table_before_mutation(tmp_path):
    store = _store(tmp_path)
    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore())
    store.commit_flux(0, 0.0, InfoStore())
    store.finalize("finished")
    output = tmp_path / "invalid_samples.csv"
    pd.DataFrame([{"dev_value": 1.0, "Freq (MHz)": 4000.0}]).to_csv(output, index=False)
    before = output.read_bytes()

    with pytest.raises(ValueError, match="missing required v2 coordinate column"):
        export_sample_table_from_artifact(store.manifest_path, filepath=output)

    assert output.read_bytes() == before


def test_controller_export_sample_table_async_uses_background_runner(tmp_path):
    app = QCoreApplication.instance()
    assert app is not None
    ctrl = build_core(project=_project(tmp_path))
    try:
        store = _store(tmp_path)
        store.write_node_row(
            "qubit_freq", 0, Patch({"qubit_freq": 5001.25}), InfoStore()
        )
        store.commit_flux(0, 0.0, InfoStore())
        store.finalize("finished")
        ctrl._last_terminal_manifest_path = store.manifest_path
        ctrl._last_terminal_status = "finished"
        output = tmp_path / "async_samples.csv"
        done: list[tuple[int, bool]] = []
        errors: list[Exception] = []

        ctrl.export_sample_table_async(
            str(output),
            on_done=lambda result: done.append(
                (result.row_count, QThread.currentThread() == app.thread())
            ),
            on_error=errors.append,
        )

        assert ctrl.quiesce_background()
        assert done == [(1, True)]
        assert errors == []
        assert output.exists()

        non_terminal = _store(tmp_path)
        non_terminal.write_node_row(
            "qubit_freq", 0, Patch({"qubit_freq": 6001.25}), InfoStore()
        )
        non_terminal.commit_flux(0, 0.5, InfoStore())
        ctrl._last_terminal_manifest_path = non_terminal.manifest_path
        ctrl._last_terminal_status = "finished"
        done.clear()

        ctrl.export_sample_table_async(
            str(tmp_path / "should_not_exist.csv"),
            on_done=lambda result: done.append(
                (result.row_count, QThread.currentThread() == app.thread())
            ),
            on_error=errors.append,
        )

        assert ctrl.quiesce_background()
        assert done == []
        assert len(errors) == 1
        assert "non-terminal" in str(errors[0])
    finally:
        ctrl.quiesce_background()

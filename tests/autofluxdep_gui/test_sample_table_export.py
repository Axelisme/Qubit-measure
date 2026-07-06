"""SampleTable export tests for autofluxdep run artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.result import QubitFreqResult
from zcu_tools.gui.app.autofluxdep.nodes.t1 import T1Builder
from zcu_tools.gui.app.autofluxdep.nodes.t2echo import T2EchoBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t2ramsey import T2RamseyBuilder
from zcu_tools.gui.app.autofluxdep.orchestrator import InfoStore
from zcu_tools.gui.app.autofluxdep.services.run_store import RunStore
from zcu_tools.gui.app.autofluxdep.services.sample_table_export import (
    export_sample_table_from_artifact,
)
from zcu_tools.gui.app.autofluxdep.state import ProjectInfo

from ._helpers import make_builder, place


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


def _store(tmp_path: Path, *, results: dict[str, object] | None = None) -> RunStore:
    nodes = [place(make_builder(name)) for name in _node_names()]
    return RunStore.create(
        project=_project(tmp_path),
        flux_values=[0.0, 0.5],
        flux_device_name="fake_flux",
        nodes=nodes,
        results=results or {},
    )


def _comment(store: RunStore) -> str:
    return f"Autofluxdep snapeshot: {store.run_dir}"


def test_export_sample_table_uses_notebook_keys_and_committed_rows(tmp_path):
    store = _store(tmp_path)
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

    store.write_node_row("qubit_freq", 1, Patch({"qubit_freq": 6001.0}), InfoStore())
    store.finalize("finished")

    result = export_sample_table_from_artifact(store.run_dir)

    assert result.row_count == 1
    df = pd.read_csv(result.path)
    assert list(df.columns) == [
        "calibrated mA",
        "Freq (MHz)",
        "T1 (us)",
        "T1err (us)",
        "T2r (us)",
        "T2r err (us)",
        "T2e (us)",
        "T2e err (us)",
        "Tcomment",
    ]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["calibrated mA"] == 0.125
    assert row["Freq (MHz)"] == 5001.25
    assert row["T1 (us)"] == 12.0
    assert row["T1err (us)"] == 0.4
    assert row["T2r (us)"] == 21.0
    assert row["T2r err (us)"] == 0.5
    assert row["T2e (us)"] == 31.0
    assert row["T2e err (us)"] == 0.6
    assert row["Tcomment"] == _comment(store)
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
                "calibrated mA": -1.0,
                "Freq (MHz)": 4000.0,
                "Tcomment": "existing",
            }
        ]
    ).to_csv(output, index=False)
    result = export_sample_table_from_artifact(store.manifest_path, filepath=output)

    assert result.path == str(output)
    assert result.row_count == 1
    df = pd.read_csv(output)
    assert len(df) == 2
    assert df.loc[0, "calibrated mA"] == -1.0
    assert df.loc[0, "Freq (MHz)"] == 4000.0
    assert df.loc[0, "Tcomment"] == "existing"
    assert df.loc[1, "calibrated mA"] == 0.0
    assert df.loc[1, "Freq (MHz)"] == 5001.25
    assert df.loc[1, "T1 (us)"] == 12.0
    assert df.loc[1, "Tcomment"] == _comment(store)


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
            "calibrated mA": 0.0,
            "Freq (MHz)": 5001.25,
            "Tcomment": _comment(store),
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
            "calibrated mA": 0.0,
            "Freq (MHz)": 5001.25,
            "Tcomment": _comment(store),
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
            "calibrated mA": 0.0,
            "Freq (MHz)": 5123.0,
            "Tcomment": _comment(store),
        }
    ]


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

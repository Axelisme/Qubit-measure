"""Independent workflow tools consume saved inputs, never journal or a live host."""

import json
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import (
    DataVariable,
    LabberPayload,
    load_grouped_labber_data,
    load_run_data,
)
from zcu_tools.resources.entry.storage import bind_sample_table

from ._independent_inputs import RUN_ID, coherence_run, mist_run
from ._sim_inputs import Inputs


def assert_channel_roundtrip(expected: LabberPayload, actual: LabberPayload) -> None:
    """Compare public-reader data, units and every inner-first axis exactly."""
    np.testing.assert_array_equal(actual.z, expected.z)
    assert (
        actual.data.name == expected.data.name
        and actual.data.unit == expected.data.unit
    )
    for got_axis, want_axis in zip(actual.axes, expected.axes, strict=True):
        assert got_axis.name == want_axis.name and got_axis.unit == want_axis.unit
        np.testing.assert_array_equal(got_axis.values, want_axis.values)


def test_combined_coherence_outputs_roundtrip_and_latest_wins(
    inputs: Inputs, tmp_path: Path
) -> None:
    from zcu_lab.workflows.coherence_export import export_run, main

    run = coherence_run(inputs, tmp_path / "run")
    # A broken journal is deliberately irrelevant to this independent tool.
    (run.paths.metadata_root / "journal.jsonl").write_text("not json")
    result, database = tmp_path / "result", tmp_path / "Database"
    report = export_run(
        run.paths.metadata_root,
        outputs=("merged", "labber", "samples"),
        result=result,
        database=database,
    )
    assert report.missing_records == ()
    assert report.samples is not None and report.samples.written_count == 3
    assert [out.status for out in report.outputs] == ["saved", "saved"]
    native = load_run_data(run.paths.data_root / "merged.h5")
    labber_path = report.outputs[1].path
    assert labber_path is not None
    labber = load_grouped_labber_data(str(labber_path))
    assert native.metadata.run_id == RUN_ID
    assert native.metadata.experiment == "coherence_fluxdep"
    assert native.metadata.snapshot == run.plan.snapshot
    assert native.metadata.provenance.git_commit == "a" * 40
    assert native.cfg is not None and native.cfg.values == run.plan.model_dump(
        mode="json"
    )
    assert native.metadata.started_at == "2026-10-07T00:00:00+00:00"
    assert set(native.payload.variables) == {
        DataVariable(k + suffix)
        for k in ("t1", "t2ramsey", "t2echo")
        for suffix in ("_signal", "_time", "")
    }
    for kind, count in (("t1", 3), ("t2ramsey", 5), ("t2echo", 2)):
        signal = native.payload.variables[DataVariable(kind + "_signal")]
        time = native.payload.variables[DataVariable(kind + "_time")]
        lifetime = native.payload.variables[DataVariable(kind)]
        assert signal.z.shape == time.z.shape == (4, count)
        assert lifetime.z.shape == (4,)
        assert [a.name for a in signal.axes] == [kind + " Time Index", "Flux"]
        assert [a.unit for a in signal.axes] == ["", "native"]
        assert time.data.unit == lifetime.data.unit == "s"
        np.testing.assert_array_equal(time.z[0], np.arange(count) * 1e-6)
        np.testing.assert_array_equal(time.z[1], np.arange(count) * 2e-6)
        assert np.isnan(time.z[2:]).all() and np.isnan(signal.z[2:]).all()
        assert lifetime.z[0] == pytest.approx((20 + count) * 1e-6)
        assert np.isnan(lifetime.z[2:]).all()
        if kind == "t2echo":
            assert np.isnan(lifetime.z[1]) and np.isfinite(signal.z[1]).all()
        for key in (kind + "_signal", kind + "_time", kind):
            expected = native.payload.variables[DataVariable(key)]
            actual = labber.variables[DataVariable(key)]
            assert_channel_roundtrip(expected, actual)
    rows = bind_sample_table(result, "Q1")
    latest = rows.latest(RUN_ID + "/1")
    assert (
        latest is not None
        and latest["dev_value"] == 61
        and latest["dev_unit"] == "native"
    )
    assert latest["freq_mhz"] == 5001 and latest["t1_us"] == 23
    assert latest["t2r_us"] == 25 and "t2e_us" not in latest
    assert latest["source"] == "workflow:" + RUN_ID and latest["point"] == "native"
    frequency_only = rows.latest(RUN_ID + "/2")
    assert (
        frequency_only is not None
        and frequency_only["freq_mhz"] == 5002
        and frequency_only["dev_value"] == 12
    )
    assert not any(k.endswith("_us") for k in frequency_only)
    assert (
        main(
            [
                str(run.paths.metadata_root),
                "--merged",
                "--labber",
                "--samples",
                "--result",
                str(result),
                "--database",
                str(database),
            ]
        )
        == 0
    )
    assert len(rows.read()) == 3 and len(rows.path.read_text().splitlines()) == 6
    assert len(list(database.rglob("*.hdf5"))) == 2


@pytest.mark.parametrize("workflow", ["coherence_fluxdep", "mist_fluxdep"])
def test_all_missing_records_names_every_iteration_before_any_write(
    inputs: Inputs, tmp_path: Path, workflow: str
) -> None:
    from zcu_lab.workflows.coherence_export import export_run as export_coherence
    from zcu_lab.workflows.mist_export import export_run as export_mist

    run = (
        coherence_run(inputs, tmp_path / "run")
        if workflow == "coherence_fluxdep"
        else mist_run(inputs, tmp_path / "run")
    )
    missing = tuple(sorted((run.paths.data_root / "iter").glob("*/record.json")))
    for path in missing:
        path.unlink()
    result, database = tmp_path / "result", tmp_path / "Database"
    if workflow == "coherence_fluxdep":
        with pytest.raises(
            ValueError, match="No (usable committed|iteration records)"
        ) as failure:
            export_coherence(
                run.paths.metadata_root,
                outputs=("merged", "labber", "samples"),
                result=result,
                database=database,
            )
    else:
        with pytest.raises(
            ValueError, match="No (usable committed|iteration records)"
        ) as failure:
            export_mist(
                run.paths.metadata_root,
                outputs=("merged", "labber"),
                database=database,
            )
    for path in missing:
        assert str(path) in str(failure.value)
    assert not (run.paths.data_root / "merged.h5").exists()
    assert not result.exists() and not database.exists()


@pytest.mark.parametrize("enabled", [(False, True, False), (True, False, True)])
def test_disabled_items_have_no_channels_or_sample_fields(
    inputs: Inputs, tmp_path: Path, enabled: tuple[bool, bool, bool]
) -> None:
    from zcu_lab.workflows.coherence_export import export_run

    run = coherence_run(
        inputs, tmp_path / "run", t1=enabled[0], ramsey=enabled[1], echo=enabled[2]
    )
    result = tmp_path / "result"
    export_run(run.paths.metadata_root, outputs=("merged", "samples"), result=result)
    stored = load_run_data(run.paths.data_root / "merged.h5")
    expected = {
        DataVariable(kind + suffix)
        for kind, on in zip(("t1", "t2ramsey", "t2echo"), enabled, strict=True)
        if on
        for suffix in ("_signal", "_time", "")
    }
    assert set(stored.payload.variables) == expected
    sample = bind_sample_table(result, "Q1").latest(RUN_ID + "/0")
    assert sample is not None
    for field, on in zip(("t1", "t2r", "t2e"), enabled, strict=True):
        assert (field + "_us" in sample) == on
    assert sample["dev_value"] == (60 if enabled[2] else 50 if enabled[1] else 40)


def test_missing_record_reports_every_iteration_and_never_recovers_native(
    inputs: Inputs, tmp_path: Path
) -> None:
    from zcu_lab.workflows.coherence_export import export_run

    run = coherence_run(inputs, tmp_path / "run")
    missing = tuple(
        run.paths.data_root / "iter" / f"{i:06d}" / "record.json" for i in (1, 3)
    )
    for path in missing:
        path.unlink()
    report = export_run(run.paths.metadata_root, outputs=("merged",))
    assert report.missing_records == missing
    stored = load_run_data(run.paths.data_root / "merged.h5")
    assert np.isnan(stored.payload.variables[DataVariable("t1_signal")].z[0]).all()


@pytest.mark.parametrize("outputs", [("merged", "samples"), ("merged", "labber")])
def test_missing_destination_refuses_before_any_write(
    inputs: Inputs, tmp_path: Path, outputs: tuple[str, ...]
) -> None:
    from zcu_lab.workflows.coherence_export import export_run

    run = coherence_run(inputs, tmp_path / "run")
    with pytest.raises(ValueError, match="result|database"):
        export_run(run.paths.metadata_root, outputs=outputs)
    assert not (run.paths.data_root / "merged.h5").exists()


@pytest.mark.parametrize(
    "corruption",
    ["json", "degraded", "identity", "plan", "index", "native", "tag", "discarded"],
)
def test_bad_saved_evidence_names_its_source_before_writes(
    inputs: Inputs, tmp_path: Path, corruption: str
) -> None:
    from zcu_lab.workflows.coherence_export import export_run

    run = coherence_run(inputs, tmp_path / "run")
    path = run.paths.data_root / "iter/000001/record.json"
    document = json.loads(path.read_text())
    if corruption == "json":
        path.write_text("{")
    elif corruption == "native":
        (path.parent / "t1.h5").unlink()
    elif corruption == "tag":
        (path.parent / "frequency-final.h5").replace(path.parent / "t1.h5")
    else:
        if corruption == "degraded":
            document["result"]["encoded_record"]["mode"] = "repr"
        elif corruption == "identity":
            document["identity"]["run_id"] = "other"
        elif corruption == "plan":
            document["plan"]["points"] = 99
        elif corruption == "index":
            document["result"]["encoded_record"]["value"]["index"] = 99
        elif corruption == "discarded":
            # The leftover files must not become commit evidence.
            document["result"] = {
                "kind": "step_discarded",
                "call_seq": 1,
                "revision": 0,
                "iteration_dir": "iter/000001",
                "reason": "pause",
            }
        path.write_text(json.dumps(document))
    if corruption == "discarded":
        report = export_run(run.paths.metadata_root, outputs=("merged",))
        assert report.outputs[0].status == "saved"
        stored = load_run_data(run.paths.data_root / "merged.h5")
        assert np.isnan(stored.payload.variables[DataVariable("t1_signal")].z[0]).all()
    else:
        with pytest.raises((ValueError, OSError), match="000001"):
            export_run(
                run.paths.metadata_root,
                outputs=("merged", "samples"),
                result=tmp_path / "result",
            )
        assert not (run.paths.data_root / "merged.h5").exists()
        assert not (tmp_path / "result").exists()


def test_mist_tool_actual_gain_and_rejects_samples(
    inputs: Inputs, tmp_path: Path
) -> None:
    from zcu_lab.workflows.coherence_export import export_run as coherence_export
    from zcu_lab.workflows.mist_export import export_run, main

    run = mist_run(inputs, tmp_path / "run")
    report = export_run(
        run.paths.metadata_root,
        outputs=("merged", "labber"),
        database=tmp_path / "Database",
    )
    assert [out.status for out in report.outputs] == ["saved", "saved"]
    native = load_run_data(run.paths.data_root / "merged.h5")
    assert set(native.payload.variables) == {
        DataVariable("signal"),
        DataVariable("success"),
    }
    signal = native.payload.variables[DataVariable("signal")]
    assert signal.z.shape == (2, 3)
    assert [(a.name, a.unit) for a in signal.axes] == [
        ("Readout Gain", "a.u."),
        ("Flux", "native"),
    ]
    np.testing.assert_array_equal(signal.axes[0].values, [0, 0.5, 1])
    np.testing.assert_array_equal(signal.z[0], [40j, 1 + 41j, 2 + 42j])
    np.testing.assert_array_equal(
        native.payload.variables[DataVariable("success")].z, [1.0, np.nan]
    )
    labber_path = report.outputs[1].path
    assert labber_path is not None
    labber = load_grouped_labber_data(str(labber_path))
    for key, expected in native.payload.variables.items():
        actual = labber.variables[key]
        assert (
            actual.z.shape == expected.z.shape
            and actual.data.unit == expected.data.unit
        )
        np.testing.assert_array_equal(actual.z, expected.z)
    with pytest.raises(ValueError, match="workflow.*coherence|coherence.*workflow"):
        coherence_export(run.paths.metadata_root, outputs=("merged",))
    with pytest.raises(SystemExit) as rejected:
        main(
            [
                str(run.paths.metadata_root),
                "--samples",
                "--result",
                str(tmp_path / "result"),
            ]
        )
    assert rejected.value.code == 2
    assert main([str(run.paths.metadata_root), "--merged"]) == 0


def test_mist_mismatched_actual_gain_fails_shared_prepare(
    inputs: Inputs, tmp_path: Path
) -> None:
    from zcu_lab.workflows.mist_export import export_run

    run = mist_run(inputs, tmp_path / "run", mismatch=True)
    with pytest.raises(ValueError, match="gain grid"):
        export_run(
            run.paths.metadata_root,
            outputs=("merged", "labber"),
            database=tmp_path / "Database",
        )
    assert (
        not (run.paths.data_root / "merged.h5").exists()
        and not (tmp_path / "Database").exists()
    )


def test_writers_fail_independently_and_keep_allocated_paths(
    inputs: Inputs, tmp_path: Path
) -> None:
    from zcu_lab.workflows.coherence_export import export_run

    run = coherence_run(inputs, tmp_path / "run")
    (run.paths.data_root / "merged.h5").mkdir()
    first = export_run(
        run.paths.metadata_root,
        outputs=("merged", "labber"),
        database=tmp_path / "Database",
    )
    assert (
        first.outputs[0].status == "failed"
        and first.outputs[0].path == run.paths.data_root / "merged.h5"
    )
    assert first.outputs[0].error and first.outputs[1].status == "saved"
    (run.paths.data_root / "merged.h5").rmdir()
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    second = export_run(
        run.paths.metadata_root, outputs=("merged", "labber"), database=blocked
    )
    assert second.outputs[0].status == "saved"
    assert second.outputs[1].status == "failed" and second.outputs[1].error

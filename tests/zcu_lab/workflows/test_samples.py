"""Science artifact and append host contracts, without live source fallback."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import JsonValue
from zcu_tools.datafile import load_run_data, save_run_data
from zcu_tools.resources.sample_table import JsonlSampleTable

from zcu_lab.workflows import export
from zcu_lab.workflows.calibration import CalibratedPi2
from zcu_lab.workflows.t1 import T1Point
from zcu_lab.workflows.t2 import T2Plan, T2Point

from ._sample_inputs import sample_engine
from ._sim_inputs import Inputs, make_plan, seed_from_inputs


@pytest.mark.parametrize("kind", ["t1", "t2echo", "t2ramsey"])
def test_samples_use_committed_scalars_and_last_native_snapshot(
    inputs: Inputs, tmp_path: Path, kind: str
) -> None:
    producer = getattr(export, f"export_{kind}_samples", None)
    assert callable(producer), "Science must expose a pure samples artifact producer"
    from zcu_tools.experiment.workflows.exporting import SampleRun

    seed = seed_from_inputs(inputs)
    base = make_plan(inputs).model_copy(update={"fluxes": (1.0, 1.1, 1.2)})
    if kind == "t1":
        engine, paths, _devices = sample_engine(
            base,
            tmp_path,
            lambda i: T1Point(
                i,
                base.fluxes[i],
                seed,
                20 + i if i < 2 else None,
                0.5 if i < 2 else None,
                None if i < 2 else "rejected",
            ),
            seed=seed,
        )
        rows = producer(
            SampleRun("sample-run", base, paths.data_root, engine.records("sample-run"))
        )
    else:
        plan = T2Plan.model_validate(base.model_dump())
        cal = CalibratedPi2(
            seed.f01_mhz,
            seed.pi_length_us,
            seed.readout_freq_mhz,
            seed.readout_gain,
            0.2,
        )
        engine, paths, _devices = sample_engine(
            plan,
            tmp_path,
            lambda i: T2Point(
                i,
                plan.fluxes[i],
                cal,
                20 + i if i < 2 else None,
                0.5 if i < 2 else None,
                None if i < 2 else "rejected",
            ),
            seed=seed,
        )
        rows = producer(
            SampleRun("sample-run", plan, paths.data_root, engine.records("sample-run"))
        )
    assert [row.key for row in rows] == ["sample-run/0", "sample-run/1"]
    assert [{field.name: field.value for field in row.fields} for row in rows] == [
        {
            "dev_value": 40.0,
            "dev_unit": "native",
            "freq_mhz": seed.f01_mhz,
            f"{kind}_us": 20.0,
            f"{kind}_err_us": 0.5,
        },
        {
            "dev_value": 41.0,
            "dev_unit": "native",
            "freq_mhz": seed.f01_mhz,
            f"{kind}_us": 21.0,
            f"{kind}_err_us": 0.5,
        },
    ]


def test_prevalidation_rejects_missing_snapshot_before_any_append(
    inputs: Inputs, tmp_path: Path
) -> None:
    producer = getattr(export, "export_t1_samples", None)
    assert callable(producer), (
        "Science must validate all candidate rows before host writes"
    )
    from zcu_tools.experiment.workflows.exporting import SampleRun
    from zcu_tools.gui.app.workflows.samples import append_samples

    plan = make_plan(inputs).model_copy(update={"fluxes": (1.0, 1.1, 1.2)})
    seed = seed_from_inputs(inputs)
    engine, paths, _devices = sample_engine(
        plan,
        tmp_path,
        lambda i: T1Point(i, plan.fluxes[i], seed, 20.0, 1.0, None),
        seed=seed,
    )
    records = engine.records("sample-run")
    last = records[-1]
    path = paths.data_root / last.iteration_dir / last.run_files[-1]
    native = load_run_data(path)
    save_run_data(
        path,
        native.payload,
        replace(native.metadata, device_snapshot=None),
        cfg=native.cfg,
        replace=True,
    )
    table = JsonlSampleTable(tmp_path / "history.jsonl")
    with pytest.raises(ValueError, match="snapshot"):
        rows = producer(SampleRun("sample-run", plan, paths.data_root, records))
        append_samples("sample-run", rows=rows, table=table, point="captured")
    assert not table.path.exists()


class InterruptedTable(JsonlSampleTable):
    """Break the actual filesystem after one public append; retain its prefix."""

    def append(
        self,
        *,
        run_id: str,
        point: str | None = None,
        source: str = "manual",
        at: datetime | None = None,
        **fields: JsonValue,
    ) -> None:
        super().append(run_id=run_id, point=point, source=source, at=at, **fields)
        if not self.path.with_suffix(".prefix").exists():
            self.path.rename(self.path.with_suffix(".prefix"))
            self.path.mkdir()


def test_partial_io_count_cause_and_retry_latest(
    inputs: Inputs, tmp_path: Path
) -> None:
    producer = getattr(export, "export_t1_samples", None)
    assert callable(producer), "Sample export must preserve partial append failures"
    from zcu_tools.experiment.workflows.exporting import SampleRun
    from zcu_tools.gui.app.workflows.samples import SampleExportError, append_samples

    plan = make_plan(inputs).model_copy(update={"fluxes": (1.0, 1.1, 1.2)})
    seed = seed_from_inputs(inputs)
    engine, paths, _devices = sample_engine(
        plan,
        tmp_path,
        lambda i: T1Point(
            i,
            plan.fluxes[i],
            seed,
            20.0 if i < 2 else None,
            1.0 if i < 2 else None,
            None if i < 2 else "rejected",
        ),
        seed=seed,
    )
    rows = producer(
        SampleRun("sample-run", plan, paths.data_root, engine.records("sample-run"))
    )
    table = InterruptedTable(tmp_path / "history.jsonl")
    with pytest.raises(SampleExportError) as failure:
        append_samples("sample-run", rows=rows, table=table, point="captured")
    error = failure.value
    assert error.written_count == 1
    assert isinstance(error.cause, IsADirectoryError) and error.__cause__ is error.cause
    assert str(error.cause) in str(error) and "1" in str(error)
    table.path.rmdir()
    table.path.with_suffix(".prefix").rename(table.path)
    assert table.read()["run_id"].tolist() == ["sample-run/0"]
    assert (
        append_samples(
            "sample-run", rows=rows, table=table, point="captured"
        ).written_count
        == 2
    )
    assert len(table.path.read_text().splitlines()) == 3
    assert table.read()["run_id"].tolist() == ["sample-run/0", "sample-run/1"]
    latest = table.latest("sample-run/1")
    assert latest is not None and latest["point"] == "captured"

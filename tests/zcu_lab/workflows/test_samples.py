"""Coherence sample preparation and append contracts, without live fallback."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter
from zcu_tools.datafile import load_run_data, save_run_data
from zcu_tools.experiment.workflows.saved_run import SavedPoint
from zcu_tools.resources.sample_table import JsonlSampleTable

from zcu_lab.workflows.calibration import FrequencyResult, RabiResult, ReadoutResult
from zcu_lab.workflows.coherence import CoherencePlan, CoherencePoint
from zcu_lab.workflows.coherence_measurements import LifetimeKind, LifetimeResult
from zcu_lab.workflows.samples import SampleExportError, append_samples, prepare_samples

from ._sample_inputs import sample_engine
from ._sim_inputs import Inputs, make_plan, seed_from_inputs


def sample_points(
    inputs: Inputs, root: Path, kind: LifetimeKind, *, all_accepted: bool = False
) -> tuple[CoherencePlan, tuple[SavedPoint[CoherencePoint], ...]]:
    """Generate reachable current points and named native acquisition snapshots."""
    seed = seed_from_inputs(inputs)
    plan = CoherencePlan.model_validate(make_plan(inputs).model_dump()).model_copy(
        update={
            "fluxes": (1.0, 1.1, 1.2),
            "measure_t1": kind == "t1",
            "measure_t2ramsey": kind == "t2ramsey",
            "measure_t2echo": kind == "t2echo",
        }
    )

    def record(i: int) -> CoherencePoint:
        accepted = i < 2 or all_accepted
        return CoherencePoint(
            i,
            plan.fluxes[i],
            FrequencyResult(
                "accepted" if accepted else "rejected",
                None if accepted else "rejected",
                seed.f01_mhz if accepted else None,
                ("frequency.h5",),
            ),
            RabiResult(
                "accepted" if accepted else "skipped",
                None if accepted else "dependency",
                seed.f01_mhz if accepted else None,
                0.4 if accepted else None,
                0.2 if accepted else None,
                "rabi.h5" if accepted else None,
            ),
            ReadoutResult("skipped", "cadence", None, None, None),
            (
                LifetimeResult(
                    kind,
                    "accepted" if accepted else "skipped",
                    None if accepted else "dependency",
                    20.0 + i if accepted else None,
                    0.5 if accepted else None,
                    "decay.h5" if accepted else None,
                ),
            ),
        )

    engine, paths, _devices = sample_engine(plan, root, record, seed=seed, kind=kind)
    points = tuple(
        SavedPoint(
            TypeAdapter(CoherencePoint).validate_python(r.encoded_record.value),
            paths.data_root / r.iteration_dir / "record.json",
        )
        for r in engine.records("sample-run")
    )
    return plan, points


@pytest.mark.parametrize("kind", ["t1", "t2echo", "t2ramsey"])
def test_samples_use_committed_scalars_and_last_native_snapshot(
    inputs: Inputs, tmp_path: Path, kind: LifetimeKind
) -> None:
    plan, points = sample_points(inputs, tmp_path, kind)
    rows = prepare_samples("sample-run", plan, points)
    field = {"t1": "t1", "t2echo": "t2e", "t2ramsey": "t2r"}[kind]
    assert [row.key for row in rows] == ["sample-run/0", "sample-run/1"]
    assert [{f.name: f.value for f in row.fields} for row in rows] == [
        {
            "dev_value": 40.0,
            "dev_unit": "native",
            "freq_mhz": seed_from_inputs(inputs).f01_mhz,
            f"{field}_us": 20.0,
            f"{field}_err_us": 0.5,
        },
        {
            "dev_value": 41.0,
            "dev_unit": "native",
            "freq_mhz": seed_from_inputs(inputs).f01_mhz,
            f"{field}_us": 21.0,
            f"{field}_err_us": 0.5,
        },
    ]


def test_prevalidation_rejects_missing_snapshot_before_any_append(
    inputs: Inputs, tmp_path: Path
) -> None:
    plan, points = sample_points(inputs, tmp_path, "t1", all_accepted=True)
    path = points[-1].directory / "decay.h5"
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
        append_samples(
            "sample-run",
            rows=prepare_samples("sample-run", plan, points),
            table=table,
            point="captured",
        )
    assert not table.path.exists()


class InterruptedTable(JsonlSampleTable):
    """Break the actual filesystem after one public append; retain its prefix."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.interrupted = False

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
        if not self.interrupted:
            self.interrupted = True
            self.path.rename(self.path.with_suffix(".prefix"))
            self.path.mkdir()


def test_partial_io_count_cause_and_retry_latest(
    inputs: Inputs, tmp_path: Path
) -> None:
    plan, points = sample_points(inputs, tmp_path, "t1")
    rows = prepare_samples("sample-run", plan, points)
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

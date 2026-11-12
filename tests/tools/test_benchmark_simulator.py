"""Benchmark orchestration and JSON comparison at importable public seams."""

from contextlib import contextmanager

import pytest
from benchmark_simulator import (
    AcquireSession,
    Workload,
    compare_results,
    main,
    run_benchmarks,
    run_workload,
)
from benchmark_simulator_workloads import make_workloads, open_session


@pytest.mark.parametrize(
    ("supported_name", "unsupported_name"),
    [
        ("single_repeated_shaped", "single_typo"),
        ("ftf_overlap_shaped", "ftf_typo"),
    ],
)
def test_open_session_rejects_unknown_name_with_supported_prefix(
    supported_name, unsupported_name
):
    workload = next(item for item in make_workloads() if item.name == supported_name)
    renamed = workload.model_copy(update={"name": unsupported_name})
    with (
        pytest.raises(ValueError, match=f"unsupported workload: {unsupported_name}"),
        open_session(renamed),
    ):
        pass


@pytest.fixture
def workload():
    return Workload(
        name="synthetic",
        version=1,
        configuration_version=1,
        configuration={"seed": 123},
    )


def test_acquire_modes_exclude_setup_and_clear_only_warm_empty_history(workload):
    wall = [0.0]
    cpu = [0.0]
    observations = []
    closed = []

    @contextmanager
    def sessions(spec):
        wall[0] += 1000
        cpu[0] += 500
        calls = [0]
        history = [0]
        seen = []
        observations.append(seen)

        def acquire():
            seen.append(history[0])
            calls[0] += 1
            history[0] += 1
            wall[0] += calls[0]
            cpu[0] += calls[0] / 2

        def clear():
            history[0] = 0
            wall[0] += 100

        try:
            yield AcquireSession(acquire, clear)
        finally:
            closed.append(True)
            wall[0] += 1000

    result = run_workload(
        workload, sessions, wall_clock=lambda: wall[0], cpu_clock=lambda: cpu[0]
    )
    assert [mode.mode for mode in result.modes] == [
        "first_acquire",
        "warm_clear_history",
        "warm_retain_history",
    ]
    assert observations == [[0]] * 5 + [[0, 1, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6]]
    assert len(closed) == 7
    first, cleared, retained = result.modes
    assert [sample.wall_seconds for sample in first.samples] == [1] * 5
    assert [sample.wall_seconds for sample in cleared.samples] == [3, 4, 5, 6, 7]
    assert cleared.wall.median_seconds == retained.wall.median_seconds == 5
    assert cleared.wall.mad_seconds == 1
    assert cleared.cpu.median_seconds == 2.5
    assert cleared.cpu.mad_seconds == 0.5
    assert len(result.workload_sha256) == len(result.configuration_sha256) == 64
    assert result.workload == workload


def test_complete_json_retains_versions_raw_samples_and_fingerprints(
    workload, tmp_path
):
    @contextmanager
    def sessions(spec):
        yield AcquireSession(lambda: None, lambda: None)

    result = run_benchmarks([workload], sessions)
    assert result.measurement["repetitions"] == 5
    assert result.measurement["warmup_acquires"] == 2
    assert result.libraries["python"]
    assert result.libraries["numpy"]
    assert len(result.workloads) == 1
    output = tmp_path / "results.json"
    assert (
        main(["--output", str(output)], workloads=[workload], session_factory=sessions)
        == 0
    )
    restored = type(result).model_validate_json(output.read_text())
    assert len(restored.workloads[0].modes) == 3
    assert all(len(mode.samples) == 5 for mode in restored.workloads[0].modes)
    altered = workload.model_copy(update={"configuration": {"seed": 124}})
    updated = run_benchmarks([altered], sessions).workloads[0]
    assert updated.configuration_sha256 != result.workloads[0].configuration_sha256
    assert updated.workload_sha256 != result.workloads[0].workload_sha256


@pytest.mark.parametrize("failure_at", [1, 6, 8, 15, 19])
def test_failed_acquire_returns_nonzero_and_never_writes_partial_result(
    workload, tmp_path, caplog, failure_at
):
    calls = [0]
    live = []

    @contextmanager
    def sessions(spec):
        live.append(True)

        def acquire():
            calls[0] += 1
            if calls[0] == failure_at:
                raise RuntimeError("synthetic acquire failed")

        try:
            yield AcquireSession(acquire, lambda: None)
        finally:
            live.pop()

    output = tmp_path / "result.json"
    output.write_text("prior evidence")
    status = main(
        ["--output", str(output)], workloads=[workload], session_factory=sessions
    )
    assert status != 0
    assert "synthetic acquire failed" in caplog.text
    assert output.read_text() == "prior evidence"
    assert live == []


@pytest.fixture
def completed_run(workload):
    @contextmanager
    def sessions(spec):
        yield AcquireSession(lambda: None, lambda: None)

    ticks = iter(range(1000))
    return run_benchmarks(
        [workload],
        sessions,
        wall_clock=lambda: float(next(ticks)),
        cpu_clock=lambda: float(next(ticks)),
    )


def test_compare_reports_both_medians_and_slower_percent_without_failing(
    completed_run, workload, tmp_path, capsys
):
    slower = completed_run.model_copy(deep=True)
    for mode in slower.workloads[0].modes:
        mode.wall.median_seconds *= 1.5
        mode.cpu.median_seconds *= 2
    rows = compare_results(completed_run, slower)
    assert len(rows) == 6
    wall, cpu = rows[:2]
    assert (wall.before_seconds, wall.after_seconds, wall.percent_change) == (
        3,
        4.5,
        50,
    )
    assert cpu.percent_change == 100
    before = tmp_path / "before.json"
    before.write_text(completed_run.model_dump_json())
    output = tmp_path / "after.json"

    @contextmanager
    def sessions(spec):
        yield AcquireSession(lambda: None, lambda: None)

    assert (
        main(
            ["--output", str(output), "--compare", str(before)],
            workloads=[workload],
            session_factory=sessions,
        )
        == 0
    )
    assert "synthetic" in capsys.readouterr().out


@pytest.mark.parametrize(
    "change",
    [
        "workload_version",
        "configuration_version",
        "configuration",
        "hash",
        "measurement",
        "schema",
        "missing_workload",
        "missing_mode",
    ],
)
def test_compare_rejects_incompatible_workload_or_measurement(completed_run, change):
    incompatible = completed_run.model_copy(deep=True)
    changed = incompatible.workloads[0]
    if change == "workload_version":
        changed.workload.version += 1
    elif change == "configuration_version":
        changed.workload.configuration_version += 1
    elif change == "configuration":
        changed.workload.configuration["seed"] = 999
    elif change == "hash":
        changed.workload_sha256 = "different"
    elif change == "measurement":
        incompatible.measurement["warmup_acquires"] = 3
    elif change == "schema":
        incompatible.schema_version += 1
    elif change == "missing_workload":
        incompatible.workloads.clear()
    else:
        changed.modes.pop()
    with pytest.raises(ValueError, match="incompatible"):
        compare_results(completed_run, incompatible)

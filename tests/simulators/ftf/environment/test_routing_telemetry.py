"""One bounded, best-effort routing record per admitted FTF acquisition."""

import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import (
    AcquireOutcome,
    SimulationBinding,
    SimulationCallbacks,
)
from zcu_tools.simulate.adc import ConditionalReadout, PreparedStatistics
from zcu_tools.simulators.ftf.environment import ftf, runtime
from zcu_tools.simulators.ftf.environment.runtime import FTFSource
from zcu_tools.simulators.ftf.physics import CavityDrive
from zcu_tools.simulators.telemetry import RoutingTelemetry

from tests.core._simulation import tick_array
from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulators._telemetry import TelemetryEngine
from tests.simulators.ftf.environment._templates import (
    herald_template,
    request,
    settings,
    snapshot,
    template,
)


@pytest.fixture(autouse=True)
def non_evolving_builder(monkeypatch):
    # ftf() has no public Engine injection; direct FTFSource callers use engine=.
    # Replace only the builder's numerical collaborator, not environment logic.
    monkeypatch.setattr(runtime, "Engine", TelemetryEngine)


@pytest.fixture
def work_clock(monkeypatch):
    clock = SimpleNamespace(
        now=0.0, evolve=0.0, readout=0.0, fail=None, fail_on=1, calls={}
    )
    # No public telemetry clock injection exists. Only public numeric collaborators
    # advance synthetic time; ADC work and the environment stay unchanged.
    # Evolution uses prescribed densities, not a solver.
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)

    def observe(method, stage, seconds):
        def run(*args, **kwargs):
            name = method.__name__
            clock.calls[name] = clock.calls.get(name, 0) + 1
            try:
                result = method(*args, **kwargs)
                if clock.fail == name and clock.calls[name] == clock.fail_on:
                    raise RuntimeError("synthetic numeric failure")
                return result
            finally:
                clock.now += seconds
                setattr(clock, stage, getattr(clock, stage) + seconds)

        return run

    monkeypatch.setattr(
        TelemetryEngine, "evolve", observe(TelemetryEngine.evolve, "evolve", 2.0)
    )
    monkeypatch.setattr(
        CavityDrive, "advance", observe(CavityDrive.advance, "readout", 3.0)
    )
    monkeypatch.setattr(
        ConditionalReadout,
        "responses",
        observe(ConditionalReadout.responses, "readout", 5.0),
    )
    monkeypatch.setattr(
        PreparedStatistics, "draw", observe(PreparedStatistics.draw, "readout", 7.0)
    )
    # There is no trace-sampler injection point; replace the environment's bound
    # public ADC function with a delegating observer, not its private logic.
    monkeypatch.setattr(
        runtime, "sample_traces", observe(runtime.sample_traces, "readout", 11.0)
    )
    return clock


@pytest.fixture
def bound_source(tmp_path):
    devices = DeviceManager()
    sources = []

    def build(*, readout=None):
        source = FTFSource(
            settings(),
            devices,
            soccfg=profile_soccfg("ftf"),
            readout=readout,
            engine=TelemetryEngine(),
            telemetry=RoutingTelemetry(tmp_path),
        )
        sources.append(source)
        snapshots = iter((snapshot(0.2), snapshot(0.3), snapshot(0.3)))
        binding = SimulationBinding(
            lambda: next(snapshots),
            source.execute,
            backend_version="synthetic",
            callbacks=SimulationCallbacks(
                on_begin=source.begin_acquire, on_end=source.finish_acquire
            ),
        )
        return source, binding

    yield build
    for source in sources:
        source.close()
    devices.close_all_devices()


def routing_rows(directory):
    return [
        json.loads(line)
        for path in directory.glob("routing-*.jsonl")
        for line in path.read_text().splitlines()
    ]


def acquire_rounds(binding, acquire, rounds):
    with binding.acquire() as sequence:
        acquire = replace(acquire, sequence=sequence)
        for index in range(rounds):
            binding.sample_round(acquire, index)


@pytest.mark.sim
@pytest.mark.parametrize("path", ["direct", "cavity", "projective"])
@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_layer_walls_sum_public_work_across_rounds_and_continuations(
    tmp_path, bound_source, work_clock, path, mode
):
    readout = None
    if path == "projective":
        plan, readout = herald_template()
    else:
        plan = template()
        if path == "cavity":
            earlier = replace(
                plan.events[0],
                start_ticks=tick_array(np.zeros(2)),
                duration_ticks=tick_array(np.full(2, 0.002)),
            )
            plan = replace(plan, events=(earlier, *plan.events))
    _, binding = bound_source(readout=readout)
    acquire_rounds(binding, request(mode, templates=(plan,)), 3)
    (record,) = routing_rows(tmp_path)
    assert record["rounds"] == 3
    assert record["complete"] is True
    assert record["compile_wall_s"] == 0.0
    assert record["materialize_wall_s"] == 0.0
    assert record["evolve_wall_s"] == work_clock.evolve
    assert record["readout_wall_s"] == work_clock.readout
    assert record["wall_s"] == work_clock.now
    with binding.acquire():
        pass
    _, empty = routing_rows(tmp_path)
    assert empty["complete"] is True
    assert empty["evolve_wall_s"] == 0.0
    assert empty["readout_wall_s"] == 0.0


@pytest.mark.sim
@pytest.mark.parametrize(
    ("failure", "mode"),
    [
        ("evolve", "integrated"),
        ("responses", "integrated"),
        ("draw", "integrated"),
        ("sample_traces", "decimated"),
    ],
)
@pytest.mark.parametrize("rounds", [1, 2])
def test_failed_numeric_stage_keeps_elapsed_work_and_writes_one_false_row(
    tmp_path, bound_source, work_clock, failure, mode, rounds
):
    _, binding = bound_source()
    work_clock.fail = failure
    work_clock.fail_on = rounds
    with pytest.raises(RuntimeError, match="synthetic numeric failure"):
        acquire_rounds(binding, request(mode), rounds)
    (record,) = routing_rows(tmp_path)
    assert record["complete"] is False
    assert record["rounds"] == rounds
    assert record["compile_wall_s"] is None
    assert record["materialize_wall_s"] is None
    assert record["evolve_wall_s"] == work_clock.evolve
    if failure == "evolve" and rounds == 1:
        assert record["readout_wall_s"] is None
    else:
        assert record["readout_wall_s"] == work_clock.readout


def test_failed_acquire_retains_measured_zero(tmp_path, bound_source, work_clock):
    _, binding = bound_source()
    with (
        pytest.raises(RuntimeError, match="compile failed"),
        binding.acquire(),
        binding.compiling(),
    ):
        raise RuntimeError("compile failed")
    (record,) = routing_rows(tmp_path)
    assert record["complete"] is False
    assert record["compile_wall_s"] == 0.0
    assert record["evolve_wall_s"] is None
    assert record["materialize_wall_s"] is None


@pytest.mark.sim
@pytest.mark.parametrize("failure", ["advance", "evolve"])
def test_projective_continuation_failure_keeps_prior_and_failed_span_work(
    tmp_path, bound_source, work_clock, failure
):
    plan, readout = herald_template()
    _, binding = bound_source(readout=readout)
    work_clock.fail = failure
    work_clock.fail_on = 2
    with pytest.raises(RuntimeError, match="synthetic numeric failure"):
        acquire_rounds(binding, request(templates=(plan,)), 1)
    (record,) = routing_rows(tmp_path)
    assert record["complete"] is False
    assert record["evolve_wall_s"] == work_clock.evolve
    assert record["readout_wall_s"] == work_clock.readout


@pytest.mark.parametrize("complete", [False, True])
def test_finish_acquire_projects_binding_outcome_without_numeric_work(
    tmp_path, complete
):
    devices = DeviceManager()
    source = FTFSource(
        settings(),
        devices,
        soccfg=profile_soccfg("ftf"),
        telemetry=RoutingTelemetry(tmp_path),
    )
    try:
        source.finish_acquire(AcquireOutcome((), 7.0, 2.0, complete, "fixture-acquire"))
        files = list(tmp_path.glob("routing-*.jsonl"))
        assert len(files) == 1
        rows = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["wall_s"] == 7.0
        assert rows[0]["compile_wall_s"] == 2.0
        assert rows[0]["complete"] is complete
        assert rows[0]["points"] == 0
        assert rows[0]["rounds"] == 0
        assert rows[0]["families"] == {}
        assert rows[0]["evolve_wall_s"] == (0.0 if complete else None)
        assert rows[0]["readout_wall_s"] == (0.0 if complete else None)
    finally:
        source.close()
        devices.close_all_devices()


@pytest.mark.sim
def test_acquire_writes_one_line_for_all_rounds_and_opt_out_is_bitwise(tmp_path: Path):
    directory = tmp_path / "enabled"
    try:
        observed = ftf(
            settings().with_telemetry(
                enabled=True, directory=directory, commit="fixture-commit", dirty=False
            ),
            soccfg=profile_soccfg("ftf"),
            time_scale=0,
        )
    except AttributeError as error:
        raise AssertionError(
            "FTF must provide configurable acquire telemetry"
        ) from error
    plain_directory = tmp_path / "disabled"
    plain = ftf(
        settings().with_telemetry(enabled=False, directory=plain_directory),
        soccfg=profile_soccfg("ftf"),
        time_scale=0,
    )
    try:
        program = compile_modules(
            observed.soccfg,
            modules=[
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=6000.0,
                    gain=0.1,
                    pre_delay=0.01,
                    trig_offset=0.02,
                    waveform=ConstWaveformCfg(length=0.04),
                    ro_length=0.04,
                ).build("ro")
            ],
            reps=2,
        )
        expected = program.acquire(
            observed.soc,
            progress=False,
            rounds=3,
        )
        actual = program.acquire(
            plain.soc,
            progress=False,
            rounds=3,
        )
        assert np.asarray(expected).tobytes() == np.asarray(actual).tobytes()
        assert not plain_directory.exists()
        files = list(directory.glob("routing-*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["points"] == 1
        assert record["rounds"] == 3
        assert record["complete"] is True
        assert record["commit"] == "fixture-commit"
        assert record["dirty"] is False
        for stage in ("compile_wall_s", "evolve_wall_s", "readout_wall_s"):
            assert type(record[stage]) is float
            assert record[stage] >= 0.0
        assert type(record["wall_s"]) is float
        assert datetime.fromisoformat(record["timestamp"]).tzinfo is not None
        assert (
            "experiment" not in record
        )  # Numeric templates carry no program identifier.
        assert sum(f["rows"] for f in record["families"].values()) == 3
        assert sum(f["cache_hits"] for f in record["families"].values()) == 2
        assert record["estimated_cost_unit"] == "relative scheduling units"
        program.acquire(
            observed.soc,
            progress=False,
            rounds=3,
        )
        records = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert len(records) == 2
        assert sum(f["cache_hits"] for f in records[1]["families"].values()) == 3
        assert sum(f["actual_cpu_s"] for f in records[1]["families"].values()) == 0.0
        assert records[1]["evolve_wall_s"] == 0.0
        assert records[1]["readout_wall_s"] >= 0.0
    finally:
        observed.devices.close_all_devices()
        plain.devices.close_all_devices()

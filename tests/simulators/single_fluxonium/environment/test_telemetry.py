"""Acquire telemetry through tiny public program/Engine/ADC seams."""

import json
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule
from zcu_tools.program.v2.modules import Module
from zcu_tools.program.v2.modules.delay import SoftDelay
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import AcquireOutcome, IntegratedRound
from zcu_tools.simulate.boundary import RoundRequest
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment import (
    acquire as acquisition,
)
from zcu_tools.simulators.single_fluxonium.environment import (
    projection,
    runtime,
    single_fluxonium,
)

from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulators._telemetry import TelemetryEngine, WorkClock
from tests.simulators.single_fluxonium.environment._inputs import settings

pytestmark = pytest.mark.sim


@pytest.fixture
def work_clock(monkeypatch):
    clock = WorkClock()
    # The environment clock controls physical pacing, not telemetry wall clocks.
    # No public telemetry timer injection exists; numeric observers keep real ADC work.
    for module in (runtime, projection, acquisition):
        monkeypatch.setattr(module, "monotonic", lambda: clock.now)
    monkeypatch.setattr(
        runtime, "prepare_readout", clock.observe(runtime.prepare_readout, 3.0)
    )
    for name in ("round_statistics", "sample_traces"):
        monkeypatch.setattr(
            acquisition, name, clock.observe(getattr(acquisition, name), 5.0)
        )
    monkeypatch.setattr(
        TelemetryEngine, "evolve", clock.observe(TelemetryEngine.evolve, 2.0)
    )
    # The environment builder has no Engine injection; Source's public injection
    # is below this program/acquire seam. Replace only its constructor collaborator.
    monkeypatch.setattr(runtime, "Engine", TelemetryEngine)
    return clock


@pytest.fixture
def telemetry_settings(tmp_path: Path):
    values = settings().model_dump(mode="json")
    values["numerics"].update(
        truncation={"native_cutoff": 6, "levels": 2},
        time_scale=0,
        projection_prune_floor=0,
        telemetry={"directory": str(tmp_path)},
    )
    return parse_single_fluxonium(values)


@pytest.fixture
def telemetry_environment(
    telemetry_settings, work_clock
) -> Generator[SimulationEnvironment]:
    env = single_fluxonium(
        telemetry_settings, soccfg=profile_soccfg("single_fluxonium")
    )
    try:
        yield env
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


@pytest.mark.parametrize("complete", [False, True])
def test_finish_projects_original_binding_outcome(
    tmp_path, telemetry_settings, complete
):
    devices = DeviceManager()
    source = runtime.SingleFluxoniumSource(telemetry_settings, devices)
    try:
        outcome = AcquireOutcome((), 7.0, 2.0, complete, "fixture-acquire")
        source.finish_acquire(outcome)
        (record,) = [
            json.loads(line)
            for path in tmp_path.glob("routing-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert record["wall_s"] == 7.0
        assert record["compile_wall_s"] == 2.0
        assert record["complete"] is complete
        assert record["evolve_wall_s"] == record["readout_wall_s"] == 0.0
    finally:
        source.close()
        devices.close_all_devices()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
@pytest.mark.parametrize("projective", [False, True], ids=["direct", "projective"])
def test_acquire_writes_one_layered_record(
    tmp_path: Path,
    projective: bool,
    mode: str,
    telemetry_environment: SimulationEnvironment,
    work_clock,
):
    env = telemetry_environment
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=7000,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.008),
        ro_length=0.004,
    )
    drive = PulseCfg(
        ch=1,
        nqz=1,
        freq=1000,
        gain=0.01,
        waveform=ConstWaveformCfg(length=0.008),
    )
    modules: list[Module] = [drive.build("mw"), readout.build("ro")]
    if projective:
        modules.extend([SoftDelay("wait", 0.016), readout.build("terminal")])
    program = compile_modules(
        env.soccfg, modules=modules, reps=2, initial_delay=0, relax_delay=0
    )
    with patch.object(
        TelemetryEngine, "evolve", autospec=True, side_effect=TelemetryEngine.evolve
    ) as evolve:
        acquire = program.acquire_decimated if mode == "decimated" else program.acquire
        assert acquire(env.soc, progress=False) is not None
        assert evolve.call_count == (3 if projective else 1)
        assert all(call.kwargs["telemetry"] is True for call in evolve.call_args_list)
    records = [
        json.loads(line)
        for path in tmp_path.glob("routing-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["complete"] is True
    assert record["points"] == record["rounds"] == 1
    assert record["extras"] == {}
    # One prefix, plus both retained states for a projective continuation.
    assert record["evolve_wall_s"] == (6.0 if projective else 2.0)
    assert record["readout_wall_s"] == 8.0  # one preparation + one ADC sample
    assert work_clock.calls["prepare_readout"] == 1
    assert record["families"]["fixture"]["rows"] == (3 if projective else 1)


@pytest.fixture
def telemetry_program(telemetry_environment: SimulationEnvironment):
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=7000,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.008),
        ro_length=0.004,
    )
    return compile_modules(
        telemetry_environment.soccfg, modules=[readout.build("ro")], reps=2
    )


@pytest.mark.parametrize("raw_path", ["projection", "round_hook"])
def test_acquire_records_raw_materialization_only_when_needed(
    tmp_path: Path,
    telemetry_environment: SimulationEnvironment,
    telemetry_program,
    raw_path: str,
):
    env = telemetry_environment
    program = telemetry_program

    def raw_hook(*_: object) -> None:
        assert program.get_raw() is not None

    if raw_path == "round_hook":
        program.acquire(env.soc, round_hook=raw_hook, progress=False, rounds=2)
    else:
        program.acquire(env.soc, threshold=0.0, progress=False, rounds=2)
    program.acquire(env.soc, progress=False)

    raw, integrated = (
        json.loads(line)
        for path in tmp_path.glob("routing-*.jsonl")
        for line in path.read_text().splitlines()
    )
    assert raw["materialize_wall_s"] > 0.0
    assert integrated["materialize_wall_s"] == 0.0


@pytest.mark.parametrize("fails", [False, True], ids=["success", "getter-error"])
def test_schedule_attributes_deferred_raw_to_its_acquire(
    tmp_path: Path,
    telemetry_environment: SimulationEnvironment,
    telemetry_program,
    fails: bool,
):
    env = telemetry_environment
    program = telemetry_program
    program.acquire(env.soc, progress=False)

    if fails:

        class FailingRaw:
            def materialize(
                self, reps: int, avg_level: int
            ) -> dict[int, NDArray[np.int64]]:
                raise RuntimeError("deferred raw source failed")

        execute = env.soc.binding.execute

        def execute_with_failed_raw(request: RoundRequest):
            result = execute(request)
            assert isinstance(result, IntegratedRound)
            return replace(result, raw=FailingRaw())

        env.soc.binding.execute = execute_with_failed_raw

    with Schedule(scope=Scope("run")) as sched:
        if fails:
            with pytest.raises(RuntimeError, match="deferred raw source failed"):
                sched.run_acquire_shots(program, env.soc, options=RunOptions(retry=2))
        else:
            shots = sched.run_acquire_shots(program, env.soc)
            assert shots is not None
            assert shots.shots[0].shape == (1, 2, 1, 2)
            program.get_raw()  # Reading the retained buffer must not publish again.

    first, last, *supplements = (
        json.loads(line)
        for path in sorted(tmp_path.glob("routing-*.jsonl"))
        for line in path.read_text().splitlines()
    )
    assert first["materialize_wall_s"] == last["materialize_wall_s"] == 0.0
    assert last["complete"] is True
    if fails:
        assert supplements == []  # Includes no reacquire on the getter error.
    else:
        (supplement,) = supplements
        assert supplement["event"] == "materialize"
        assert supplement["acquire_id"] == last["acquire_id"] != first["acquire_id"]
        assert supplement["materialize_wall_s"] > 0.0


@pytest.mark.parametrize("seam", ["evolve", "continuation", "adc"])
def test_numeric_failure_still_writes_incomplete_record(
    tmp_path: Path,
    telemetry_environment: SimulationEnvironment,
    seam: str,
    work_clock,
):
    env = telemetry_environment
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=7000,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.008),
        ro_length=0.004,
    )
    modules: list[Module] = [readout.build("ro")]
    if seam == "continuation":
        modules.extend([SoftDelay("wait", 0.016), readout.build("terminal")])
    program = compile_modules(env.soccfg, modules=modules, reps=2)
    work_clock.fail = "round_statistics" if seam == "adc" else "evolve"
    work_clock.fail_on = 2 if seam == "continuation" else 1
    with pytest.raises(RuntimeError, match="synthetic numeric failure"):
        program.acquire(env.soc, progress=False)
    records = [
        json.loads(line)
        for path in tmp_path.glob("routing-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["complete"] is False
    assert record["rounds"] == 1
    assert record["evolve_wall_s"] == (4.0 if seam == "continuation" else 2.0)
    assert record["readout_wall_s"] == (8.0 if seam == "adc" else 0.0)
    assert record["families"].get("fixture", {}).get("rows", 0) == (
        0 if seam == "evolve" else 1
    )


def test_writer_failure_warns_once_without_changing_acquire(
    tmp_path: Path,
    telemetry_environment: SimulationEnvironment,
):
    env = telemetry_environment
    # Make the configured directory unusable without patching the writer.
    for path in tmp_path.iterdir():
        path.unlink()
    tmp_path.rmdir()
    tmp_path.write_text("not a directory")
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=7000,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.008),
        ro_length=0.004,
    )
    program = compile_modules(env.soccfg, modules=[readout.build("ro")], reps=2)
    with pytest.warns(
        RuntimeWarning, match="routing telemetry could not be written"
    ) as warnings:
        assert program.acquire(env.soc, progress=False) is not None
        assert program.acquire(env.soc, progress=False) is not None
    # Cold dependency imports can also emit unrelated warnings during acquire.
    routing_warnings = [
        warning
        for warning in warnings
        if issubclass(warning.category, RuntimeWarning)
        and str(warning.message).startswith("routing telemetry could not be written")
    ]
    assert len(routing_warnings) == 1

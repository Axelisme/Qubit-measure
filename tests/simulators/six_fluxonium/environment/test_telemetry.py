"""Six-source telemetry stays at the acquire boundary, including routed spans."""

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import (
    MuxReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import AcquireOutcome
from zcu_tools.simulators.single_fluxonium.environment import (
    projection,
)
from zcu_tools.simulators.single_fluxonium.environment import (
    runtime as single_runtime,
)
from zcu_tools.simulators.six_fluxonium.config import parse_six_fluxonium
from zcu_tools.simulators.six_fluxonium.environment import runtime, six_fluxonium

from tests.program._profiles import profile_soccfg
from tests.simulators._telemetry import TelemetryEngine, WorkClock
from tests.simulators.six_fluxonium.config._data import six_document

pytestmark = pytest.mark.sim


@pytest.fixture
def work_clock(monkeypatch):
    clock = WorkClock()
    # Physical SimulationClock does not control the telemetry timers. There is
    # no public timer injection; observers delegate all real readout computation.
    for module in (runtime, single_runtime, projection):
        monkeypatch.setattr(module, "monotonic", lambda: clock.now)
    monkeypatch.setattr(
        single_runtime,
        "prepare_readout",
        clock.observe(single_runtime.prepare_readout, 3.0),
    )
    for name in ("round_statistics", "sample_traces"):
        monkeypatch.setattr(runtime, name, clock.observe(getattr(runtime, name), 5.0))
    monkeypatch.setattr(
        TelemetryEngine, "evolve", clock.observe(TelemetryEngine.evolve, 2.0)
    )

    # Six's builder/source has no public Engine injection. Replace only this
    # out-of-scope numerical collaborator, leaving six real sources and ADCs.
    def engine(*args, **kwargs):
        answer = TelemetryEngine(*args, **kwargs)
        answer.reuse_allowed = False  # every round performs known work
        return answer

    monkeypatch.setattr(runtime, "Engine", engine)
    return clock


@pytest.fixture
def telemetry_settings(tmp_path: Path):
    values = parse_six_fluxonium(six_document()).model_dump(mode="json")
    values["numerics"]["telemetry"] = {"directory": str(tmp_path)}
    for qubit in values["qubits"].values():
        qubit["numerics"]["truncation"] = {"native_cutoff": 6, "levels": 2}
        qubit["numerics"]["projection_prune_floor"] = 0
        qubit["chip"]["temperature_k"] = 0.05
        qubit["chip"]["qubit"]["q_cap"] = 3000
    return parse_six_fluxonium(values)


@pytest.fixture
def telemetry_environment(
    telemetry_settings, work_clock
) -> Generator[SimulationEnvironment]:
    env = six_fluxonium(telemetry_settings, soccfg=profile_soccfg("six_fluxonium"))
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
    source = runtime.SixFluxoniumSource(telemetry_settings, devices)
    try:
        outcome = AcquireOutcome((), 7.0, 2.0, complete)
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
@pytest.mark.parametrize("scenario", ["routed", "projective", "failure"])
def test_six_sources_write_one_acquire_record(
    tmp_path: Path,
    scenario: str,
    mode: str,
    telemetry_environment: SimulationEnvironment,
    work_clock,
):
    env = telemetry_environment
    projective = scenario == "projective"
    failed = scenario == "failure"
    builder = TimelineBuilder(env.soccfg, reps=2, initial_delay=0.016, relax_delay=0)
    if projective:
        readout = MuxReadoutCfg(
            gen_ch=0,
            nqz=1,
            mixer_freq=7100,
            length=0.064,
            ro_length=0.026,
            tones=[
                MuxReadoutTone(ro_ch=ch, freq=6800 + 125 * ch, gain=0.1)
                for ch in range(6)
            ],
        ).build("mux")
        with builder.repeat("reads", 2):
            builder.emit(readout)
            builder.timeline_inc(0.096)
    else:
        frequencies = [
            6800 + 125 * ch + offset for ch in range(6) for offset in (-5, 5)
        ]
        freq = builder.add_table_sweep("frequency", frequencies)
        readout = PulseReadoutCfg(
            gen_ch=13,
            ro_ch=6,
            nqz=1,
            freq=6800,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.016),
            ro_length=0.004,
        )
        builder.emit(PulseReadout("pulse", readout, freq=freq))
    builder.delay_by_timeline()
    program = builder.build()
    if projective:
        program.set_reads_per_shot(2)
    acquire = program.acquire_decimated if mode == "decimated" else program.acquire
    if failed:
        work_clock.fail, work_clock.fail_on = "evolve", 4
    # The explicit fixture diagnostics expose lost/duplicated source reports.
    with patch.object(
        TelemetryEngine, "evolve", autospec=True, side_effect=TelemetryEngine.evolve
    ) as evolve:
        if failed:
            with pytest.raises(RuntimeError, match="synthetic numeric failure"):
                acquire(env.soc, progress=False, rounds=3)
        else:
            assert acquire(env.soc, progress=False, rounds=3) is not None
            assert len({id(call.args[0]) for call in evolve.call_args_list}) == 6
            assert evolve.call_count == (54 if projective else 18)
        assert all(call.kwargs["telemetry"] is True for call in evolve.call_args_list)
    records = [
        json.loads(line)
        for path in tmp_path.glob("routing-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["complete"] is not failed
    assert record["rounds"] == (1 if failed else 3)
    assert record["points"] == (1 if projective else 12)
    assert record["extras"] == {}
    # Six sources x three rounds; projective spans are prefix + two states.
    assert record["evolve_wall_s"] == (8.0 if failed else 108.0 if projective else 36.0)
    assert record["readout_wall_s"] == (24.0 if failed else 144.0)
    assert record["families"]["fixture"]["rows"] == (
        3 if failed else 54 if projective else 18
    )
    if not failed:
        assert acquire(env.soc, progress=False, rounds=3) is not None
        repeated = [
            json.loads(line)
            for path in tmp_path.glob("routing-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert len(repeated) == 2
        assert repeated[1]["rounds"] == 3
        assert repeated[1]["evolve_wall_s"] == record["evolve_wall_s"]
        assert repeated[1]["readout_wall_s"] == 144.0
        assert (
            repeated[1]["families"]["fixture"]["rows"]
            == record["families"]["fixture"]["rows"]
        )

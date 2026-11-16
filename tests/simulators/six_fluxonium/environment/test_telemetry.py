"""Six-source telemetry stays at the acquire boundary, including routed spans."""

import json
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import (
    MuxReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.simulate.evolution import Engine, FamilyRouting, PathDiagnostics
from zcu_tools.simulators.six_fluxonium.config import parse_six_fluxonium
from zcu_tools.simulators.six_fluxonium.environment import six_fluxonium

from tests.program._profiles import profile_soccfg
from tests.simulators.six_fluxonium.config._data import six_document

pytestmark = pytest.mark.sim


@pytest.fixture
def telemetry_environment(tmp_path: Path) -> Generator[SimulationEnvironment]:
    values = parse_six_fluxonium(six_document()).model_dump(mode="json")
    values["numerics"]["telemetry"] = {"directory": str(tmp_path)}
    for qubit in values["qubits"].values():
        qubit["numerics"]["truncation"] = {"native_cutoff": 6, "levels": 2}
        qubit["chip"]["temperature_k"] = 0.05
        qubit["chip"]["qubit"]["q_cap"] = 3000
    env = six_fluxonium(
        parse_six_fluxonium(values), soccfg=profile_soccfg("six_fluxonium")
    )
    try:
        yield env
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
@pytest.mark.parametrize("scenario", ["routed", "projective", "failure"])
def test_six_sources_write_one_acquire_record(
    tmp_path: Path,
    scenario: str,
    mode: str,
    telemetry_environment: SimulationEnvironment,
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
    original = Engine.evolve
    # Distinct known seam diagnostics expose lost/duplicated source reports.
    with patch.object(Engine, "evolve", autospec=True) as evolve:

        def evolve_with_report(*args, **kwargs):
            if failed and evolve.call_count == 4:
                raise RuntimeError("fourth source failed")
            return replace(
                original(*args, **kwargs),
                paths=PathDiagnostics(
                    routing={"fixture": FamilyRouting(rows=1)},
                    segments_by_kernel={"fixture": 1},
                    duration_ns_by_kernel={"fixture": 1.0},
                ),
            )

        evolve.side_effect = evolve_with_report
        if failed:
            with pytest.raises(RuntimeError, match="fourth source failed"):
                acquire(env.soc, progress=False)
        else:
            assert acquire(env.soc, progress=False) is not None
            assert len({id(call.args[0]) for call in evolve.call_args_list}) == 6
            assert evolve.call_count >= (12 if projective else 6)
        assert all(call.kwargs["telemetry"] is True for call in evolve.call_args_list)
    records = [
        json.loads(line)
        for path in tmp_path.glob("routing-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["complete"] is not failed
    assert record["rounds"] == 1
    assert record["points"] == (1 if projective else 12)
    assert record["extras"] == {}
    for stage in ("wall_s", "compile_wall_s", "evolve_wall_s", "readout_wall_s"):
        assert isinstance(record[stage], float)
    assert record["families"]["fixture"]["rows"] == (3 if failed else evolve.call_count)
    if not failed:
        assert acquire(env.soc, progress=False) is not None
        repeated = [
            json.loads(line)
            for path in tmp_path.glob("routing-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert len(repeated) == 2
        assert repeated[1]["rounds"] == 1
        assert (
            repeated[1]["families"]["fixture"]["rows"]
            == record["families"]["fixture"]["rows"]
        )

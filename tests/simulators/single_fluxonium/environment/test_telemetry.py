"""Acquire telemetry through tiny public program/Engine/ADC seams."""

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.program.v2.modules import Module
from zcu_tools.program.v2.modules.delay import SoftDelay
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.simulate.evolution import Engine
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment import single_fluxonium

from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulators.single_fluxonium.environment._inputs import settings

pytestmark = pytest.mark.sim


@pytest.fixture
def telemetry_environment(tmp_path: Path) -> Generator[SimulationEnvironment]:
    values = settings().model_dump(mode="json")
    values["numerics"].update(
        truncation={"native_cutoff": 6, "levels": 2},
        time_scale=0,
        telemetry={"directory": str(tmp_path)},
    )
    env = single_fluxonium(
        parse_single_fluxonium(values), soccfg=profile_soccfg("single_fluxonium")
    )
    try:
        yield env
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
@pytest.mark.parametrize("projective", [False, True], ids=["direct", "projective"])
def test_acquire_writes_one_layered_record(
    tmp_path: Path,
    projective: bool,
    mode: str,
    telemetry_environment: SimulationEnvironment,
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
        Engine, "evolve", autospec=True, side_effect=Engine.evolve
    ) as evolve:
        acquire = program.acquire_decimated if mode == "decimated" else program.acquire
        assert acquire(env.soc, progress=False) is not None
        assert evolve.call_count > (1 if projective else 0)
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
    for stage in ("wall_s", "compile_wall_s", "evolve_wall_s", "readout_wall_s"):
        assert isinstance(record[stage], float)
    assert sum(family["rows"] for family in record["families"].values()) > 0


@pytest.mark.parametrize("seam", ["evolve", "adc"])
def test_numeric_failure_still_writes_incomplete_record(
    tmp_path: Path,
    telemetry_environment: SimulationEnvironment,
    seam: str,
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
    program = compile_modules(env.soccfg, modules=[readout.build("ro")], reps=2)
    target = (
        "zcu_tools.simulate.evolution.Engine.evolve"
        if seam == "evolve"
        else "zcu_tools.simulators.single_fluxonium.environment.acquire.round_statistics"
    )
    with (
        patch(target, side_effect=RuntimeError("numeric failure")),
        pytest.raises(RuntimeError, match="numeric failure"),
    ):
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
    for stage in ("compile_wall_s", "evolve_wall_s", "readout_wall_s"):
        assert isinstance(record[stage], float)
    if seam == "evolve":
        assert record["readout_wall_s"] == 0.0


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
    assert len(warnings) == 1

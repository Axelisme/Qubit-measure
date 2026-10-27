"""Log-loss drift contracts with supplied times and independent RNG streams."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.core.simulation import AcquireRequest, ReadoutWindow
from zcu_tools.experiment.single_fluxonium import (
    SingleFluxoniumRuntimeOptions,
    SingleFluxoniumSource,
    replay_round,
    single_fluxonium,
)
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import (
    IntegratedRound,
    SimulationBinding,
    SimulationClock,
)
from zcu_tools.resources.simulation_replay import read_replay, replay_paths
from zcu_tools.resources.single_fluxonium import (
    SingleFluxoniumSettings,
    parse_single_fluxonium,
)

from ._inputs import settings, template


def drift_settings(sigma: float | None = 0.3):
    document = settings().model_dump(mode="json")
    document["chip"]["temperature_k"] = 0.05
    document["chip"]["qubit"].update(q_cap=1e6, xqp=1e-7, aphi=1e-6)
    document["numerics"]["time_scale"] = 0.0
    document["numerics"]["truncation"] = {"native_cutoff": 6, "levels": 2}
    document["drift"] = (
        None
        if sigma is None
        else {
            name: {"sigma": sigma, "tau_s": 10.0} for name in ("q_cap", "xqp", "aphi")
        }
    )
    return parse_single_fluxonium(document)


def request_for(binding: SimulationBinding, sequence: int):
    compiled = template()
    readout = replace(
        compiled.events[-1],
        phase_deg=compiled.events[-1].phase_deg.astype(float),
        frequency_mhz=compiled.events[-1].frequency_mhz.astype(float),
    )
    window = replace(
        compiled.readouts[0],
        frequency_mhz=compiled.readouts[0].frequency_mhz.astype(float),
    )
    compiled = replace(
        compiled, points=np.array([0, 1]), events=(readout,), readouts=(window,)
    )
    return AcquireRequest(
        "integrated",
        (8, 2),
        0,
        (ReadoutWindow(0, 0, 4, 1000, (0.018,)),),
        binding.root_seed,
        sequence,
        binding.rng_version,
        point_dims=(2,),
        point_names=("sweep",),
        reps=8,
        templates=(compiled,),
    )


@contextmanager
def environment(config, root: Path, clock: SimulationClock, *, archive: bool = False):
    env = single_fluxonium(
        config,
        clock=clock,
        truth_log=root / "truth.jsonl",
        private_replay=root / "replay" if archive else None,
    )
    try:
        yield env
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def integrated(result) -> IntegratedRound:
    if not isinstance(result, IntegratedRound):
        raise AssertionError("integrated acquisition must return statistics and raw")
    return result


@pytest.mark.sim
def test_acquisition_freezes_loss_at_admission_and_replays_after_drift(tmp_path: Path):
    now = [100.0]
    clock = SimulationClock(monotonic=lambda: now[0])
    with environment(drift_settings(), tmp_path, clock, archive=True) as env:
        binding = env.soc.binding
        results = []
        for sequence in range(2):
            binding.begin()
            admitted_time = now[0]
            now[0] += 20.0  # Request construction must not move the sampling time.
            try:
                request = request_for(binding, sequence)
                results.append(integrated(binding.sample_round(request, 0)))
                binding.sample_round(request, 1)
            finally:
                binding.end()
            truth = tmp_path / "truth.jsonl"
            assert truth.exists(), (
                "admission must write simulation-owned acquisition truth"
            )
            rows = [json.loads(line) for line in truth.read_text().splitlines()]
            assert len(rows) == sequence + 1
            assert rows[-1]["sim_now"] == pytest.approx(admitted_time)
        rows = [
            json.loads(line)
            for line in (tmp_path / "truth.jsonl").read_text().splitlines()
        ]
        assert rows[0]["q_cap"] == 1e6
        assert rows[1]["session"] == rows[0]["session"]
        for name in ("q_cap", "xqp", "aphi"):
            assert rows[1][name] != rows[0][name]
        archives = [
            (path, read_replay(path)) for path in replay_paths(tmp_path / "replay")
        ]
        assert len(archives) == 4
        for _, saved in archives:
            archived_settings = saved.acquire.settings
            if not isinstance(archived_settings, SingleFluxoniumSettings):
                pytest.fail("single-model replay must retain single-model settings")
            assert archived_settings.drift is None
            assert (
                archived_settings.chip.qubit.q_cap
                == rows[saved.acquire.request.sequence]["q_cap"]
            )
        archived = next(
            path
            for path, saved in archives
            if saved.acquire.request.sequence == 0 and saved.round.index == 0
        )
        replayed = integrated(replay_round(archived))
        np.testing.assert_array_equal(
            replayed.statistics.channels[0].sums, results[0].statistics.channels[0].sums
        )
        np.testing.assert_array_equal(
            replayed.raw.materialize(8, 0)[0], results[0].raw.materialize(8, 0)[0]
        )


@pytest.mark.sim
def test_direct_source_samples_once_per_acquisition_and_changes_physical_rates(
    tmp_path: Path,
):
    now = [0.0]
    clock = SimulationClock(monotonic=lambda: now[0])
    config = drift_settings()
    with environment(config, tmp_path / "env", clock) as env:
        source = SingleFluxoniumSource(
            config,
            env.devices,
            runtime=SingleFluxoniumRuntimeOptions(
                clock=clock, truth_log=tmp_path / "direct.jsonl"
            ),
        )
        try:
            request = request_for(env.soc.binding, 0)
            source.open_acquire(request)
            first = (
                source.model_facts(source.snapshot())
                .source.model((0,))
                .rates_per_ns.copy()
            )
            first_settings = source.effective_settings
            now[0] += 100
            assert source.effective_settings == first_settings
            source.open_acquire(replace(request, sequence=1))
            second = (
                source.model_facts(source.snapshot()).source.model((0,)).rates_per_ns
            )
            assert not np.array_equal(first, second)
            rows = [
                json.loads(line)
                for line in (tmp_path / "direct.jsonl").read_text().splitlines()
            ]
            assert [row["sim_now"] for row in rows] == [0.0, 100.0]
        finally:
            source.close()


@pytest.mark.sim
def test_drift_stream_does_not_change_sampling_and_disabled_drift_writes_nothing(
    tmp_path: Path,
):
    now = [0.0]
    clock = SimulationClock(monotonic=lambda: now[0])
    raw = []
    for name, sigma in (("static", None), ("drifting", 0.3)):
        config = drift_settings(sigma)
        document = config.model_dump(mode="json")
        document["chip"]["temperature_k"] = 0.0
        document["chip"]["qubit"].update(xqp=0.0, aphi=0.0)
        document["drift"] = (
            None if sigma is None else {"q_cap": {"sigma": sigma, "tau_s": 10.0}}
        )
        with environment(
            parse_single_fluxonium(document), tmp_path / name, clock
        ) as env:
            program = ModularProgramV2(
                env.soccfg,
                reps=8,
                modules=[
                    PulseReadoutCfg(
                        gen_ch=0,
                        ro_ch=0,
                        nqz=1,
                        freq=7000.0,
                        gain=0.1,
                        waveform=ConstWaveformCfg(length=0.024),
                        ro_length=0.004,
                    ).build("readout")
                ],
            )
            per_acquire = []
            for _ in range(2):
                program.acquire(env.soc, rounds=2, progress=False)
                per_acquire.append(np.asarray(program.get_raw()).copy())
                now[0] += 100
            raw.append(per_acquire)
    np.testing.assert_array_equal(raw[0], raw[1])
    assert not (tmp_path / "static" / "truth.jsonl").exists()
    rows = [
        json.loads(line)
        for line in (tmp_path / "drifting" / "truth.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["q_cap"] != rows[1]["q_cap"]


def test_truth_io_failure_releases_binding_admission(tmp_path: Path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied")
    clock = SimulationClock(monotonic=lambda: 0.0)
    with environment(drift_settings(), blocked, clock) as env:
        for _ in range(2):
            with pytest.raises(FileExistsError):
                env.soc.binding.begin()

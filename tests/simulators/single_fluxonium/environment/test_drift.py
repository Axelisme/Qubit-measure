"""Log-loss drift contracts with supplied times and independent RNG streams."""

import gzip
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import (
    SimulationBinding,
    SimulationClock,
)
from zcu_tools.simulate.boundary import AcquireRequest, ReadoutWindow
from zcu_tools.simulate.evolution import capture_directory
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment import (
    SingleFluxoniumRuntimeOptions,
    SingleFluxoniumSource,
    single_fluxonium,
)

from tests.core._simulation import TIME_BASE, tick_tuple
from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulators.single_fluxonium.environment._inputs import settings, template


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
        (ReadoutWindow(0, 0, 4, TIME_BASE.period_ticks(1000), tick_tuple((0.018,))),),
        binding.root_seed,
        sequence,
        binding.rng_version,
        point_dims=(2,),
        point_names=("sweep",),
        reps=8,
        templates=(compiled,),
        time_base=TIME_BASE,
    )


@contextmanager
def environment(config, root: Path, clock: SimulationClock):
    env = single_fluxonium(
        config,
        soccfg=profile_soccfg("single_fluxonium"),
        clock=clock,
        truth_log=root / "truth.jsonl",
    )
    try:
        yield env
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


@pytest.mark.sim
def test_acquisition_freezes_loss_at_admission_and_dumps_effective_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path / "capture-cache"))
    now = [100.0]
    clock = SimulationClock(monotonic=lambda: now[0])
    document = drift_settings().model_dump(mode="json")
    document["numerics"]["debug_capture"] = True
    with environment(parse_single_fluxonium(document), tmp_path, clock) as env:
        binding = env.soc.binding
        for sequence in range(2):
            with binding.acquire() as admitted_sequence:
                assert admitted_sequence == sequence
                admitted_time = now[0]
                now[0] += 20.0  # Request construction must not move the sampling time.
                request = request_for(binding, sequence)
                binding.sample_round(request, 0)
                binding.sample_round(request, 1)
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
    captures = list(capture_directory().glob("capture-*.json.gz"))
    assert len(captures) == 2
    observed = {}
    for capture in captures:
        with gzip.open(capture, "rt", encoding="utf-8") as stream:
            metadata = json.load(stream)["metadata"]
        effective = metadata["effective_settings"]
        assert effective["drift"] is None
        observed[metadata["sampling"]["sequence"]] = {
            name: effective["chip"]["qubit"][name] for name in ("q_cap", "xqp", "aphi")
        }
    assert observed == {
        sequence: {name: row[name] for name in ("q_cap", "xqp", "aphi")}
        for sequence, row in enumerate(rows)
    }


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
            program = compile_modules(
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
            with pytest.raises(FileExistsError), env.soc.binding.acquire():
                pytest.fail("truth I/O must fail before entering acquire")

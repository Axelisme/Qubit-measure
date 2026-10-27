"""Independent acquisition-frozen coherence losses through six-source seams."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.core.simulation import AcquireRequest, ReadoutWindow, RoundRequest
from zcu_tools.experiment.six_fluxonium import six_fluxonium
from zcu_tools.experiment.six_fluxonium.runtime import SixFluxoniumSource
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg
from zcu_tools.program.v2.simulation import SimulationClock
from zcu_tools.resources.six_fluxonium import parse_six_fluxonium

from tests.experiment.single_fluxonium._inputs import slot, template
from tests.resources._six_fluxonium_data import six_document


def drift_configuration():
    document = six_document()
    for qubit in document["qubits"].values():
        qubit["chip"]["temperature_k"] = 0.04
        qubit["chip"]["qubit"].update(q_cap=1e6, xqp=1e-7, aphi=1e-6)
        qubit["drift"] = {
            name: {"sigma": 0.3, "tau_s": 10.0} for name in ("q_cap", "xqp", "aphi")
        }
    return parse_six_fluxonium(document)


@contextmanager
def environment(configuration, *, clock=None, truth_log=None):
    env = six_fluxonium(configuration, clock=clock, truth_log=truth_log)
    try:
        yield env
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_admission_writes_per_qubit_truth_with_one_session_and_time(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    with environment(drift_configuration()) as env:
        binding = env.soc.binding
        binding.begin()
        binding.end()
        logs = list(Path(tmp_path).rglob("truth.jsonl"))
        assert len(logs) == 1, "six admission must write one environment truth log"
        rows = [json.loads(line) for line in logs[0].read_text().splitlines()]
        assert [row["qubit"] for row in rows] == [f"Q{i}" for i in range(1, 7)]
        assert len({row["session"] for row in rows}) == 1
        assert len({row["sim_now"] for row in rows}) == 1
        assert all(row["q_cap"] == 1e6 for row in rows)


def test_independent_streams_rebuild_from_root_seed_and_time_sequence(tmp_path):
    config = drift_configuration()
    sequences = []
    for run in range(2):
        now = [0.0]
        clock = SimulationClock(monotonic=lambda now=now: now[0])
        truth = tmp_path / f"run-{run}.jsonl"
        document = config.model_dump(mode="json")
        if run:
            document["qubits"] = dict(reversed(document["qubits"].items()))
            document["qubits"]["Q2"]["numerics"]["seed"] = 9876
        with environment(
            parse_six_fluxonium(document), clock=clock, truth_log=truth
        ) as env:
            for now[0] in (0.0, 3.0, 25.0):
                env.soc.binding.begin()
                env.soc.binding.end()
        rows = [json.loads(line) for line in truth.read_text().splitlines()]
        sequences.append(
            {name: [row[name] for row in rows] for name in ("q_cap", "xqp", "aphi")}
        )
        for name in sequences[-1]:
            q1 = sequences[-1][name][0::6]
            q2 = sequences[-1][name][1::6]
            assert q1[0] == q2[0]
            assert q1[1:] != q2[1:]
            assert q1[1] != q1[0]
        assert len({row["session"] for row in rows}) == 1
    assert sequences[0] == sequences[1]


def routed_request(binding, sequence):
    compiled = template()
    windows = tuple(
        replace(
            compiled.readouts[0],
            channel=channel,
            frequency_mhz=np.full((2, 1), frequency, dtype=float),
        )
        for channel, frequency in enumerate((6800, 6925))
    )
    compiled = replace(
        compiled,
        points=np.array([0, 1]),
        events=tuple(
            slot(0, 16, 8, frequency_mhz=float(frequency), gain=0.1)
            for frequency in (6800, 6925)
        ),
        readouts=windows,
    )
    return AcquireRequest(
        "integrated",
        (4, 2),
        0,
        tuple(ReadoutWindow(ch, 0, 4, 1000, (0.018,)) for ch in range(2)),
        binding.root_seed,
        sequence,
        binding.rng_version,
        point_dims=(2,),
        point_names=("sweep",),
        reps=4,
        templates=(compiled,),
        relax_delay_us=125000,
    )


@pytest.mark.sim
def test_execute_routes_frozen_losses_across_points_and_rounds(tmp_path):
    now = [100.0]
    clock = SimulationClock(monotonic=lambda: now[0])
    config = drift_configuration()
    truth = tmp_path / "truth.jsonl"
    with environment(config, clock=clock, truth_log=tmp_path / "unused") as env:
        source = SixFluxoniumSource(config, env.devices, clock=clock, truth_log=truth)
        try:
            requests = [routed_request(env.soc.binding, seq) for seq in range(2)]
            acquisitions = []
            for request in requests:
                source.begin_acquire(seed=request.root_seed)
                rows = [json.loads(line) for line in truth.read_text().splitlines()]
                frozen = {row["qubit"]: row for row in rows[-6:]}
                acquisitions.append(frozen)
                for index in range(2):
                    now[0] += 10.0
                    source.execute(RoundRequest(request, source.snapshot(), index))
                    for cavity in (0, 1):
                        losses = source.sources[cavity].effective_settings.chip.qubit
                        for name in ("q_cap", "xqp", "aphi"):
                            assert (
                                getattr(losses, name) == frozen[f"Q{cavity + 1}"][name]
                            )
                assert len(truth.read_text().splitlines()) == 6 * (request.sequence + 1)
            for name in ("q_cap", "xqp", "aphi"):
                assert acquisitions[0]["Q1"][name] != acquisitions[1]["Q1"][name]
        finally:
            source.close()


@pytest.mark.sim
def test_outer_round_credits_physical_duration_once_for_all_qubits(tmp_path):
    clock = SimulationClock(monotonic=lambda: 100.0)
    truth = tmp_path / "truth.jsonl"
    with environment(drift_configuration(), clock=clock, truth_log=truth) as env:
        binding = env.soc.binding
        sequence = binding.begin()
        request = routed_request(binding, sequence)
        duration = 4 * 2 * (125000 + 0.024) / 1e6
        try:
            for index in range(2):
                binding.sample_round(request, index)
                assert clock.now() == pytest.approx(100.0 + (index + 1) * duration)
        finally:
            binding.end()
        binding.begin()
        binding.end()
        rows = [json.loads(line) for line in truth.read_text().splitlines()]
        assert [row["sim_now"] for row in rows[:6]] == [100.0] * 6
        assert [row["sim_now"] for row in rows[6:]] == pytest.approx(
            [100.0 + 2 * duration] * 6
        )


@pytest.mark.sim
@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_zero_amplitude_drift_preserves_sampling_and_drift_off_writes_nothing(
    tmp_path, mode
):
    baseline = drift_configuration().model_dump(mode="json")
    outputs = []
    for drift in (None, "omitted", "zero"):
        document = json.loads(json.dumps(baseline))
        for qubit in document["qubits"].values():
            if drift is None:
                qubit["drift"] = None
            elif drift == "omitted":
                del qubit["drift"]
            else:
                for parameter in qubit["drift"].values():
                    parameter["sigma"] = 0
        truth = tmp_path / f"{drift}.jsonl"
        clock = SimulationClock(monotonic=lambda: 10.0)
        with environment(
            parse_six_fluxonium(document), clock=clock, truth_log=truth
        ) as env:
            readout = MuxReadoutCfg(
                gen_ch=0,
                nqz=1,
                mixer_freq=7100,
                length=0.064,
                ro_length=0.026,
                tones=[
                    MuxReadoutTone(ro_ch=ch, freq=freq, gain=0.1)
                    for ch, freq in enumerate((6800.0, 6925.0))
                ],
            )
            program = ModularProgramV2(
                env.soccfg, modules=[readout.build("readout")], reps=4
            )
            if mode == "integrated":
                average = program.acquire(env.soc, rounds=2, progress=False)
                outputs.append((average, program.get_raw()))
            else:
                outputs.append(
                    (program.acquire_decimated(env.soc, rounds=2, progress=False),)
                )
        assert truth.exists() == (drift == "zero")
    for candidate in outputs[1:]:
        for expected_group, actual_group in zip(outputs[0], candidate, strict=True):
            for expected, actual in zip(expected_group, actual_group, strict=True):
                np.testing.assert_array_equal(expected, actual)

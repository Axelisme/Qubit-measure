"""Acquisition-frozen FTF loss streams through public source and binding seams."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from zcu_tools.core.simulation import RoundSignals
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.ftf import FTFAcquisitionOptions, FTFSource, ftf
from zcu_tools.program.v2.simulation import SimulationClock
from zcu_tools.resources.ftf import parse_ftf

from tests.experiment._ftf_templates import (
    expect_statistics,
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


@pytest.fixture(autouse=True)
def bounded_blas_threads():
    # These small matrix seams do not benefit from serial-run BLAS fan-out.
    with threadpool_limits(limits=1, user_api="blas"):
        yield


def drift_settings(*, sigma: float = 0.3):
    document = settings().model_dump(mode="json")
    for loop in ("f1", "coupler", "f2"):
        document["chip"][loop].update(q_cap=1e6, aphi=1e-6)
        if loop != "coupler":
            document["chip"][loop].update(xqp=1e-7, q_ind=1e8)
    document["drift"] = {
        loop: {
            name: {"sigma": sigma, "tau_s": 10.0}
            for name in (
                ("q_cap", "aphi") if loop == "coupler" else ("q_cap", "xqp", "aphi")
            )
        }
        for loop in ("f1", "coupler", "f2")
    }
    return parse_ftf(document).with_telemetry(enabled=False)


@contextmanager
def source_at(configuration, now, truth):
    source = FTFSource(
        configuration,
        DeviceManager(),
        acquisition=FTFAcquisitionOptions(
            clock=SimulationClock(monotonic=lambda: now[0]), truth_log=truth
        ),
    )
    try:
        yield source
    finally:
        source.close()


def test_loop_streams_rebuild_from_seed_and_time_and_log_nullable_squid_losses(
    tmp_path: Path,
):
    configuration = drift_settings()
    sequences = []
    sessions = []
    for run in range(2):
        now = [100.0]
        truth = tmp_path / f"run-{run}.jsonl"
        with source_at(configuration, now, truth) as source:
            sequence = []
            identities = []
            for now[0] in (100.0, 103.0, 125.0):
                source.begin_acquire(seed=123)
                effective = source.effective_settings
                assert effective.drift is None
                identities.append(source.identity)
                sequence.append(effective.chip)
            assert len(set(identities)) == 3
            rows = [json.loads(line) for line in truth.read_text().splitlines()]
            assert [row["qubit"] for row in rows] == ["f1", "coupler", "f2"] * 3
            assert [row["sim_now"] for row in rows] == [100.0] * 3 + [103.0] * 3 + [
                125.0
            ] * 3
            assert len({row["session"] for row in rows}) == 1
            sessions.append(rows[0]["session"])
            assert all(row["wall_time"].endswith("+00:00") for row in rows)
            assert all(
                row["q_ind"] is None and row["xqp"] is None for row in rows[1::3]
            )
            for index, chip in enumerate(sequence):
                for offset, loop in enumerate(("f1", "coupler", "f2")):
                    for name in (
                        ("q_cap", "aphi")
                        if loop == "coupler"
                        else ("q_cap", "q_ind", "xqp", "aphi")
                    ):
                        assert rows[index * 3 + offset][name] == getattr(
                            getattr(chip, loop), name
                        )
            assert sequence[0] == configuration.chip
            for loop in ("f1", "coupler", "f2"):
                for name in (
                    ("q_cap", "aphi") if loop == "coupler" else ("q_cap", "xqp", "aphi")
                ):
                    assert getattr(getattr(sequence[1], loop), name) != getattr(
                        getattr(sequence[0], loop), name
                    )
            assert (
                len(
                    {
                        getattr(sequence[1], loop).q_cap
                        for loop in ("f1", "coupler", "f2")
                    }
                )
                == 3
            )
            sequences.append(sequence)
    assert sequences[0] == sequences[1]
    assert sessions[0] != sessions[1]


def test_begin_invalidates_previous_session_before_new_request(tmp_path: Path):
    now = [100.0]
    with source_at(drift_settings(), now, tmp_path / "truth.jsonl") as source:
        old = source.open_acquire(request())
        now[0] += 10.0
        source.begin_acquire(seed=123)
        with pytest.raises(RuntimeError, match="no longer active"):
            old.sample_round(snapshot(), 0)


def test_acquisition_freezes_losses_and_rebuilds_rates_at_fixed_flux(tmp_path: Path):
    now = [100.0]
    truth = tmp_path / "truth.jsonl"
    with source_at(drift_settings(), now, truth) as source:
        rates = []
        for acquisition_index in range(2):
            source.begin_acquire(seed=123)
            frozen = source.effective_settings
            rates.append(
                source.model_facts(snapshot())
                .joint.rates(np.array([0.3, 0.22, 0.4]))
                .copy()
            )
            now[0] += 10.0
            acquire = request(templates=(template(tones=1),))
            session = source.open_acquire(acquire)
            with source_at(frozen, now, tmp_path / "disabled.jsonl") as reference:
                expected = reference.open_acquire(acquire)
                for index in range(2):
                    now[0] += 10.0
                    actual_result = expect_statistics(
                        session.sample_round(snapshot(), index)
                    )
                    expected_result = expect_statistics(
                        expected.sample_round(snapshot(), index)
                    )
                    np.testing.assert_array_equal(
                        actual_result.channels[0].sums, expected_result.channels[0].sums
                    )
                    assert source.effective_settings == frozen
            assert len(truth.read_text().splitlines()) == (acquisition_index + 1) * 3
        assert not np.array_equal(rates[0], rates[1])
        assert not (tmp_path / "disabled.jsonl").exists()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_disabled_and_zero_amplitude_drift_preserve_every_sample(tmp_path: Path, mode):
    configuration = drift_settings(sigma=0)
    outputs = []
    for drift in ("omitted", None, configuration.drift):
        document = configuration.model_dump(mode="json")
        if drift == "omitted":
            del document["drift"]
        else:
            document["drift"] = None if drift is None else drift.model_dump(mode="json")
        now = [100.0]
        truth = tmp_path / f"{len(outputs)}.jsonl"
        with source_at(parse_ftf(document), now, truth) as source:
            session = source.open_acquire(request(mode))
            result = session.sample_round(snapshot(), 0)
            if mode == "integrated":
                statistics = expect_statistics(result)
                arrays = [
                    statistics.state_counts,
                    *(channel.sums for channel in statistics.channels),
                ]
            else:
                if not isinstance(result, RoundSignals):
                    raise AssertionError(
                        "decimated acquisition must return channel traces"
                    )
                arrays = [channel.iq for channel in result.channels]
            outputs.append(arrays)
        assert truth.exists() == (drift is not None and drift != "omitted")
    for output in outputs[1:]:
        for expected, actual in zip(outputs[0], output, strict=True):
            np.testing.assert_array_equal(expected, actual)


def test_builder_admission_and_rounds_share_one_physical_clock(tmp_path: Path):
    now = [100.0]
    clock = SimulationClock(monotonic=lambda: now[0])
    truth = tmp_path / "truth.jsonl"
    env = ftf(
        drift_settings(),
        time_scale=0,
        acquisition=FTFAcquisitionOptions(clock=clock, truth_log=truth),
    )
    try:
        binding = env.soc.binding
        sequence = binding.begin()
        assert len(truth.read_text().splitlines()) == 3
        acquire = replace(request(), sequence=sequence, relax_delay_us=125000)
        duration = 3 * 2 * (125000 + 0.030) / 1e6
        try:
            for index in range(2):
                binding.sample_round(acquire, index)
                assert clock.now() == pytest.approx(100 + (index + 1) * duration)
        finally:
            binding.end()
        assert len(truth.read_text().splitlines()) == 3
        binding.begin()
        binding.end()
        rows = [json.loads(line) for line in truth.read_text().splitlines()]
        assert [row["sim_now"] for row in rows[3:]] == pytest.approx(
            [100 + 2 * duration] * 3
        )
        assert rows[0]["q_cap"] != rows[3]["q_cap"]
        assert binding.rounds == ()
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()

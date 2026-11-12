"""Public fixed-scenario construction and reproducible manifest freezing."""

from contextlib import closing

import numpy as np
import pytest

from tools.family_pareto_inputs import Input, canonical_json, freeze_manifest
from tools.family_pareto_scenarios import make_real_inputs


@pytest.fixture(scope="module")
def fixed_inputs() -> list[Input]:
    return make_real_inputs()


def test_fixed_scenarios_freeze_byte_identically_from_fresh_construction() -> None:
    first_inputs = make_real_inputs()
    assert first_inputs, "fixed-scenario construction must yield usable numeric inputs"
    first = canonical_json(freeze_manifest(first_inputs)) + "\n"
    second = canonical_json(freeze_manifest(make_real_inputs())) + "\n"
    assert first.encode() == second.encode()


def test_gain_scan_compiles_distinct_amplitudes_with_fixed_carrier() -> None:
    from tools import family_pareto_scenarios

    item = family_pareto_scenarios.gain_sweep_input()
    with closing((case := item.build()).source):
        scans = [
            track
            for track in case.batch.tracks
            if np.all(track.active)
            and np.ptp(track.amplitude.real) > 0
            and np.ptp(track.frequency_ghz) == 0
        ]
    assert scans, "a gain scan must compile to distinct amplitudes at fixed carrier"
    assert all(len(track.amplitude) == 24 for track in scans)


def test_length_scan_compiles_monotone_distinct_pulse_supports(
    fixed_inputs: list[Input],
) -> None:
    durations = []
    for item in fixed_inputs:
        with closing((case := item.build()).source):
            durations.extend(
                track.duration_ns
                for track in case.batch.tracks
                if np.all(track.active) and np.ptp(track.duration_ns) > 0
            )
    assert durations, "a length scan must change compiled pulse support"
    assert all(
        len(values) == 24 and np.all(np.diff(values) > 0) for values in durations
    )


def test_idle_scan_changes_gap_between_fixed_gates(fixed_inputs: list[Input]) -> None:
    gaps = []
    for item in fixed_inputs:
        with closing((case := item.build()).source):
            tracks = case.batch.tracks
            if len(tracks) != 2 or not all(np.all(t.active) for t in tracks):
                continue
            gap = tracks[1].start_ns - tracks[0].start_ns - tracks[0].duration_ns
            if np.ptp(gap) > 0:
                gaps.append(gap)
                assert all(np.ptp(t.duration_ns) == 0 for t in tracks)
    assert gaps, "an idle scan must vary a gap, not the gates' durations"
    assert all(len(gap) == 24 and np.all(np.diff(gap) > 0) for gap in gaps)


def test_gate_scan_preserves_variable_sequence_lengths(
    fixed_inputs: list[Input],
) -> None:
    inventories = []
    for item in fixed_inputs:
        with closing((case := item.build()).source):
            count = np.sum([t.active for t in case.batch.tracks], axis=0)
            if np.ptp(count) > 0:
                inventories.append(count)
                assert len(np.unique(count)) == 24
                assert np.all(np.diff(case.batch.end_ns) > 0)
    assert inventories, "a gate scan must preserve ragged, varying pulse counts"


def test_shaped_gate_scan_compiles_gaussian_phase_alphabet_prefixes() -> None:
    from tools import family_pareto_scenarios

    item = family_pareto_scenarios.shaped_gate_sequence_input()
    with closing((case := item.build()).source):
        batch = case.batch
        assert len(batch.points) == 24
        for point, count in enumerate(range(1, 25)):
            gates = sorted(
                (track for track in batch.tracks if track.active[point]),
                key=lambda track: track.start_ns[point],
            )
            assert len(gates) == count
            np.testing.assert_allclose(
                [gate.phase_rad[point] for gate in gates],
                np.resize(np.arange(4) * np.pi / 2, count),
                atol=1e-12,
            )
            envelopes = [gate.envelopes[gate.envelope_index[point]] for gate in gates]
            assert all(envelope.kind == "sampled" for envelope in envelopes)
            assert all(np.ptp(envelope.samples.real) > 0 for envelope in envelopes)
        assert np.all(np.diff(batch.end_ns) > 0)


def test_joint_scan_compiles_synchronous_independent_drive_ports(
    fixed_inputs: list[Input],
) -> None:
    synchronous = []
    for item in fixed_inputs:
        with closing((case := item.build()).source):
            tracks = case.batch.tracks
            if len(tracks) != 2 or tracks[0].port == tracks[1].port:
                continue
            first, second = tracks
            assert np.all(first.active) and np.all(second.active)
            np.testing.assert_array_equal(first.start_ns, second.start_ns)
            np.testing.assert_array_equal(first.duration_ns, second.duration_ns)
            assert len(case.batch.points) == 24
            level = tuple(int(value) for value in case.batch.flux.level[0])
            model = case.source.model(level)
            ports = {drive.name: drive for drive in model.drives}
            assert not np.allclose(
                ports[first.port].in_phase_ghz, ports[second.port].in_phase_ghz
            ), "the simultaneous ports must drive independent joint-model operators"
            assert np.all(case.batch.end_ns >= first.start_ns + first.duration_ns)
            synchronous.append(item.name)
    assert synchronous, "a joint scan must preserve two simultaneous physical ports"

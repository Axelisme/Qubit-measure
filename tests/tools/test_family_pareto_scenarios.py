"""Public fixed-scenario construction and reproducible manifest freezing."""

from contextlib import closing

import numpy as np

from tools.family_pareto_inputs import canonical_json, freeze_manifest
from tools.family_pareto_scenarios import make_real_inputs


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


def test_length_scan_compiles_monotone_distinct_pulse_supports() -> None:
    from tools.family_pareto_scenarios import length_sweep_input

    durations = []
    for item in [length_sweep_input()]:
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


def test_idle_scan_changes_gap_between_fixed_gates() -> None:
    from tools.family_pareto_scenarios import idle_sweep_input

    gaps = []
    for item in [idle_sweep_input()]:
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


def test_gate_scan_preserves_variable_sequence_lengths() -> None:
    from tools.family_pareto_scenarios import gate_sequence_input

    item = gate_sequence_input()
    with closing((case := item.build()).source):
        count = np.sum([track.active for track in case.batch.tracks], axis=0)
        np.testing.assert_array_equal(count, np.arange(1, 25))
        assert np.all(np.diff(case.batch.end_ns) > 0)


def test_long_constant_gate_scan_compiles_original_phase_alphabet() -> None:
    from tools import family_pareto_scenarios

    item = family_pareto_scenarios.long_gate_sequence_input()
    original = family_pareto_scenarios.gate_sequence_input()
    with (
        closing((case := item.build()).source),
        closing((short := original.build()).source),
    ):
        batch = case.batch
        assert len(batch.points) == 3
        np.testing.assert_array_equal(
            np.sum([track.active for track in batch.tracks], axis=0),
            [50, 100, 200],
        )
        np.testing.assert_array_equal(
            case.source.model((0,)).energies_ghz,
            short.source.model((0,)).energies_ghz,
        )
        np.testing.assert_array_equal(
            case.source.initial_density, short.source.initial_density
        )
        prefix = sorted(
            (track for track in short.batch.tracks if track.active[-1]),
            key=lambda track: track.start_ns[-1],
        )
        for point, count in enumerate((50, 100, 200)):
            gates = sorted(
                (track for track in batch.tracks if track.active[point]),
                key=lambda track: track.start_ns[point],
            )
            np.testing.assert_allclose(
                [gate.phase_rad[point] for gate in gates],
                np.resize([0, np.pi / 2, np.pi, 3 * np.pi / 2], count),
                atol=1e-12,
            )
            assert all(
                gate.envelopes[gate.envelope_index[point]].kind == "constant"
                for gate in gates
            )
            np.testing.assert_allclose(
                [gate.duration_ns[point] for gate in gates], 20, atol=2
            )
            for gate, reference in zip(gates[: len(prefix)], prefix, strict=True):
                assert gate.port == reference.port
                for field in (
                    "start_ns",
                    "duration_ns",
                    "frequency_ghz",
                    "phase_rad",
                    "amplitude",
                ):
                    np.testing.assert_array_equal(
                        getattr(gate, field)[point], getattr(reference, field)[-1]
                    )
            gaps = [
                second.start_ns[point]
                - first.start_ns[point]
                - first.duration_ns[point]
                for first, second in zip(gates[:-1], gates[1:], strict=True)
            ]
            assert np.all(np.asarray(gaps) >= 0)
            assert np.any(np.asarray(gaps) > 0)
            assert (
                batch.end_ns[point]
                > gates[-1].start_ns[point] + gates[-1].duration_ns[point]
            )
        assert np.all(np.diff(batch.end_ns) > 0)


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


def test_long_shaped_gate_scan_compiles_original_gaussian_alphabet() -> None:
    from tools import family_pareto_scenarios

    item = family_pareto_scenarios.long_shaped_gate_sequence_input()
    original = family_pareto_scenarios.shaped_gate_sequence_input()
    with (
        closing((case := item.build()).source),
        closing((short := original.build()).source),
    ):
        batch = case.batch
        assert len(batch.points) == 3
        np.testing.assert_array_equal(
            np.sum([track.active for track in batch.tracks], axis=0),
            [50, 100, 200],
        )
        np.testing.assert_array_equal(
            case.source.model((0,)).energies_ghz,
            short.source.model((0,)).energies_ghz,
        )
        np.testing.assert_array_equal(
            case.source.initial_density, short.source.initial_density
        )
        prefix = sorted(
            (track for track in short.batch.tracks if track.active[-1]),
            key=lambda track: track.start_ns[-1],
        )
        reference_envelope = prefix[0].envelopes[prefix[0].envelope_index[-1]]
        for point, count in enumerate((50, 100, 200)):
            gates = sorted(
                (track for track in batch.tracks if track.active[point]),
                key=lambda track: track.start_ns[point],
            )
            np.testing.assert_allclose(
                [gate.phase_rad[point] for gate in gates],
                np.resize([0, np.pi / 2, np.pi, 3 * np.pi / 2], count),
                atol=1e-12,
            )
            for gate in gates:
                envelope = gate.envelopes[gate.envelope_index[point]]
                assert envelope.kind == "sampled"
                assert np.ptp(envelope.samples.real) > 0
                np.testing.assert_array_equal(
                    envelope.samples, reference_envelope.samples
                )
                np.testing.assert_array_equal(
                    gate.amplitude[point], prefix[0].amplitude[-1]
                )
                np.testing.assert_array_equal(
                    gate.frequency_ghz[point], prefix[0].frequency_ghz[-1]
                )
            np.testing.assert_allclose(
                [gate.duration_ns[point] for gate in gates], 100, atol=2
            )
            for gate, reference in zip(gates[: len(prefix)], prefix, strict=True):
                assert gate.port == reference.port
                for field in (
                    "start_ns",
                    "duration_ns",
                    "frequency_ghz",
                    "phase_rad",
                    "amplitude",
                ):
                    np.testing.assert_array_equal(
                        getattr(gate, field)[point], getattr(reference, field)[-1]
                    )
            assert all(
                second.start_ns[point]
                >= first.start_ns[point] + first.duration_ns[point]
                for first, second in zip(gates[:-1], gates[1:], strict=True)
            )
            assert (
                batch.end_ns[point]
                > gates[-1].start_ns[point] + gates[-1].duration_ns[point]
            )
        assert np.all(np.diff(batch.end_ns) > 0)


def test_delay_scan_compiles_two_same_qubit_gaussians_from_separated_to_overlapping() -> (
    None
):
    from tools import family_pareto_scenarios

    item = family_pareto_scenarios.overlapping_shaped_tones_input()
    with closing((case := item.build()).source):
        batch = case.batch
        assert len(batch.points) == 24
        assert len(batch.tracks) == 2
        first, second = sorted(batch.tracks, key=lambda track: track.start_ns[0])
        assert np.all(first.active) and np.all(second.active)
        assert first.start_ns[0] + first.duration_ns[0] <= second.start_ns[0]
        assert first.start_ns[-1] == second.start_ns[-1]
        assert first.duration_ns[-1] == second.duration_ns[-1]
        lead = second.start_ns - first.start_ns
        assert np.all(np.diff(lead) < 0)
        assert np.allclose(first.duration_ns, 300, atol=2)
        for track in batch.tracks:
            envelope = track.envelopes[track.envelope_index[0]]
            assert envelope.kind == "sampled" and np.ptp(envelope.samples.real) > 0
        model = case.source.model((0,))
        assert len(model.energies_ghz) == 20
        ports = {drive.name: drive for drive in model.drives}
        assert first.port != second.port
        np.testing.assert_array_equal(
            ports[first.port].in_phase_ghz, ports[second.port].in_phase_ghz
        )


def test_constant_tone_scan_compiles_fully_overlapping_same_qubit_drives() -> None:
    from tools import family_pareto_scenarios

    item = family_pareto_scenarios.overlapping_constant_tones_input()
    with closing((case := item.build()).source):
        batch = case.batch
        assert len(batch.points) == 24
        assert len(batch.tracks) == 2
        first, second = batch.tracks
        assert first.port != second.port
        assert np.all(first.active) and np.all(second.active)
        np.testing.assert_array_equal(first.start_ns, second.start_ns)
        np.testing.assert_array_equal(first.duration_ns, second.duration_ns)
        assert np.all(first.duration_ns > 0)
        for track in batch.tracks:
            assert all(envelope.kind == "constant" for envelope in track.envelopes)
        carrier_ranges = sorted(np.ptp(track.frequency_ghz) for track in batch.tracks)
        assert carrier_ranges[0] == 0 and carrier_ranges[1] > 0
        scanned = max(batch.tracks, key=lambda track: np.ptp(track.frequency_ghz))
        assert np.all(np.diff(scanned.frequency_ghz) > 0)
        model = case.source.model((0,))
        ports = {drive.name: drive for drive in model.drives}
        np.testing.assert_array_equal(
            ports[first.port].in_phase_ghz, ports[second.port].in_phase_ghz
        )
        assert np.any(ports[first.port].in_phase_ghz != 0)
        assert np.all(batch.end_ns >= first.start_ns + first.duration_ns)


def test_joint_scan_compiles_synchronous_independent_drive_ports() -> None:
    from tools.family_pareto_scenarios import simultaneous_drive_input

    synchronous = []
    for item in [simultaneous_drive_input()]:
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

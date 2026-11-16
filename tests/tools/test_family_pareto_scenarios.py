"""Public fixed-scenario construction and reproducible manifest freezing."""

from collections.abc import Iterator
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.program import build_soccfg
from zcu_tools.program.v2 import Pulse, TimelineBuilder
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.simulate.evolution import SecularModel
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment import (
    SingleFluxoniumSource,
    single_fluxonium,
)

from tools import family_pareto_scenarios as scene_producer
from tools.family_pareto_inputs import canonical_json, freeze_manifest, make_inputs
from tools.family_pareto_scenarios import make_real_inputs
from tools.family_pareto_scene_builders import single_model_input
from tools.family_pareto_settings import BOARD_PROFILE, SINGLE_MODEL


def test_producer_main_freezes_numeric_inputs_without_measurement(
    tmp_path: Path,
) -> None:
    assert callable(getattr(scene_producer, "main", None)), (
        "missing scene producer entry point"
    )
    from dataclasses import replace

    from zcu_tools.simulate.evolution import read_frozen_manifest, read_physical_input

    from tools import family_pareto_inputs

    assert callable(getattr(family_pareto_inputs, "freeze_records", None)), (
        "missing typed frozen scene producer"
    )
    declarations = make_inputs()
    declarations.append(replace(declarations[0], name="caller-real", category="real"))
    assert scene_producer.main(["--output", str(tmp_path)], inputs=declarations) == 0
    synthetic = read_frozen_manifest(tmp_path / "frozen-synthetic.json.gz")
    real = read_frozen_manifest(tmp_path / "frozen-real.json.gz")
    assert [record.name for record in synthetic] == ["idle_coherence"]
    assert [record.name for record in real] == ["caller-real"]
    frozen = read_physical_input(synthetic[0].physical)
    with closing(frozen.source):
        np.testing.assert_array_equal(frozen.batch.points, [0, 1])
        assert frozen.source.initial_density.shape == (3, 3)


@pytest.fixture(scope="module")
def bias045_model() -> Iterator[SecularModel]:
    settings = parse_single_fluxonium(SINGLE_MODEL)
    settings = settings.model_copy(
        update={
            "dc_bias": settings.dc_bias.model_copy(update={"offset_phi0": 0.45}),
            "drift": None,
        }
    )
    env = single_fluxonium(settings, soccfg=build_soccfg(BOARD_PROFILE))
    with ExitStack() as stack:
        stack.callback(env.devices.close_all_devices)
        if env.close_backend is not None:
            stack.callback(env.close_backend)
        owner = stack.enter_context(
            closing(SingleFluxoniumSource(settings, env.devices))
        )
        facts = owner.model_facts(owner.snapshot())
        stack.enter_context(closing(facts.source))
        assert facts.model.dc_flux == pytest.approx(0.45)
        yield facts.source.model((0,))


def assert_same_model(actual: SecularModel, expected: SecularModel) -> None:
    assert actual.energies_ghz.shape == (21,)
    np.testing.assert_allclose(actual.energies_ghz, expected.energies_ghz)
    np.testing.assert_allclose(actual.rates_per_ns, expected.rates_per_ns)
    np.testing.assert_allclose(actual.dephasing_per_ns, expected.dephasing_per_ns)
    assert len(actual.drives) == len(expected.drives)
    for drive, expected_drive in zip(actual.drives, expected.drives, strict=True):
        assert drive.name == expected_drive.name
        np.testing.assert_allclose(drive.in_phase_ghz, expected_drive.in_phase_ghz)
        np.testing.assert_allclose(drive.quadrature_ghz, expected_drive.quadrature_ghz)


def test_bias045_carrier_scan_uses_biased_model_and_preserves_compiled_offsets(
    bias045_model: SecularModel,
) -> None:
    from tools import family_pareto_scenarios

    item = family_pareto_scenarios.carrier_sweep_bias045_input()
    original = family_pareto_scenarios.carrier_sweep_input()
    with (
        closing((case := item.build()).source),
        closing((base := original.build()).source),
    ):
        model = case.source.model((0,))
        base_model = base.source.model((0,))
        assert_same_model(model, bias045_model)
        assert not np.allclose(model.energies_ghz, base_model.energies_ghz)
        center = model.energies_ghz[1] - model.energies_ghz[0]
        base_center = base_model.energies_ghz[1] - base_model.energies_ghz[0]
        assert len(case.batch.points) == 24
        assert len(case.batch.tracks) == len(base.batch.tracks) == 1
        track, base_track = case.batch.tracks[0], base.batch.tracks[0]
        np.testing.assert_allclose(
            track.frequency_ghz - center,
            base_track.frequency_ghz - base_center,
            atol=1e-8,
        )
        assert np.ptp(track.frequency_ghz) == pytest.approx(
            np.ptp(base_track.frequency_ghz)
        )
        np.testing.assert_array_equal(track.active, base_track.active)
        np.testing.assert_array_equal(track.start_ns, base_track.start_ns)
        np.testing.assert_array_equal(track.duration_ns, base_track.duration_ns)
        np.testing.assert_array_equal(track.amplitude, base_track.amplitude)
        np.testing.assert_array_equal(track.phase_rad, base_track.phase_rad)
        assert all(envelope.kind == "constant" for envelope in track.envelopes)
        np.testing.assert_array_equal(track.envelope_index, base_track.envelope_index)
        np.testing.assert_array_equal(case.batch.end_ns, base.batch.end_ns)
        np.testing.assert_array_equal(case.batch.flux.level, base.batch.flux.level)


def test_bias045_gate_scan_uses_biased_model_and_preserves_compiled_prefixes(
    bias045_model: SecularModel,
) -> None:
    from tools import family_pareto_scenarios

    item = family_pareto_scenarios.gate_sequence_bias045_input()
    original = family_pareto_scenarios.gate_sequence_input()
    with (
        closing((case := item.build()).source),
        closing((base := original.build()).source),
    ):
        model = case.source.model((0,))
        assert_same_model(model, bias045_model)
        assert not np.allclose(model.energies_ghz, base.source.model((0,)).energies_ghz)
        center = model.energies_ghz[1] - model.energies_ghz[0]
        batch = case.batch
        np.testing.assert_array_equal(
            np.sum([track.active for track in batch.tracks], axis=0),
            np.arange(1, 25),
        )
        assert len(batch.tracks) == len(base.batch.tracks)
        for track, base_track in zip(batch.tracks, base.batch.tracks, strict=True):
            np.testing.assert_allclose(
                track.frequency_ghz[track.active], center, atol=1e-8
            )
            np.testing.assert_array_equal(track.active, base_track.active)
            np.testing.assert_array_equal(track.start_ns, base_track.start_ns)
            np.testing.assert_array_equal(track.duration_ns, base_track.duration_ns)
            np.testing.assert_array_equal(track.amplitude, base_track.amplitude)
            np.testing.assert_array_equal(track.phase_rad, base_track.phase_rad)
            assert all(envelope.kind == "constant" for envelope in track.envelopes)
            np.testing.assert_array_equal(
                track.envelope_index, base_track.envelope_index
            )
        for point, count in enumerate(range(1, 25)):
            gates = sorted(
                (track for track in batch.tracks if track.active[point]),
                key=lambda track: track.start_ns[point],
            )
            np.testing.assert_allclose(
                [gate.phase_rad[point] for gate in gates],
                np.resize(np.arange(4) * np.pi / 2, count),
                atol=1e-12,
            )
        np.testing.assert_array_equal(batch.end_ns, base.batch.end_ns)
        np.testing.assert_array_equal(batch.flux.level, base.batch.flux.level)


@pytest.mark.parametrize(
    ("scenario", "original_name"),
    [
        ("random_gate_sequence", "gate_sequence"),
        ("random_shaped_gate_sequence", "shaped_gate_sequence"),
    ],
)
def test_random_gate_scan_preserves_model_pulses_and_companions(
    scenario: str, original_name: str
) -> None:
    (item,) = make_real_inputs(scenario)
    (original,) = make_real_inputs(original_name)
    assert item.companions == original.companions
    with (
        closing((case := item.build()).source),
        closing((reference := original.build()).source),
    ):
        assert_same_model(case.source.model((0,)), reference.source.model((0,)))
        np.testing.assert_array_equal(
            case.source.initial_density, reference.source.initial_density
        )
        assert case.families == reference.families
        batch = case.batch
        assert len(batch.points) == 8
        counts = np.sum([track.active for track in batch.tracks], axis=0)
        assert np.unique(counts).size > 1
        assert counts[0] == 0
        # User-authored seed 4091952314, depth 32: 43 primitives plus one recovery.
        selected_point = 1
        selected = sorted(
            (track for track in batch.tracks if track.active[selected_point]),
            key=lambda track: track.start_ns[selected_point],
        )
        assert len(selected) == 44
        np.testing.assert_allclose(
            [gate.phase_rad[selected_point] for gate in selected[:12]],
            np.deg2rad([90, 90, 0, 90, 0, 90, 0, 90, 0, 270, 0, 90]),
            atol=1e-12,
        )
        assert selected[-1].phase_rad[selected_point] == 0
        exemplar = next(track for track in reference.batch.tracks if track.active[-1])
        # The compiler can add a clock step to support at a different absolute start.
        allowed_durations = np.unique(
            np.concatenate(
                [track.duration_ns[track.active] for track in reference.batch.tracks]
            )
        )
        for point in range(len(batch.points)):
            gates = sorted(
                (track for track in batch.tracks if track.active[point]),
                key=lambda track: track.start_ns[point],
            )
            for gate in gates:
                assert gate.duration_ns[point] in allowed_durations
                for field in ("frequency_ghz", "amplitude"):
                    np.testing.assert_array_equal(
                        getattr(gate, field)[point], getattr(exemplar, field)[-1]
                    )
                actual = gate.envelopes[gate.envelope_index[point]]
                expected = exemplar.envelopes[exemplar.envelope_index[-1]]
                assert actual.kind == expected.kind
                np.testing.assert_array_equal(actual.samples, expected.samples)
            if gates:
                # Readout-start clips generator support rounded on a different clock.
                assert batch.end_ns[point] > gates[-1].start_ns[point]


@pytest.mark.parametrize(
    "scenario", ["random_gate_sequence", "random_shaped_gate_sequence"]
)
def test_random_gate_scan_freezes_byte_identically(scenario: str) -> None:
    first = canonical_json(freeze_manifest(make_real_inputs(scenario))) + "\n"
    second = canonical_json(freeze_manifest(make_real_inputs(scenario))) + "\n"
    assert first.encode() == second.encode()


def test_fixed_scenarios_freeze_byte_identically_from_fresh_construction() -> None:
    first_inputs = make_real_inputs()
    assert first_inputs, "fixed-scenario construction must yield usable numeric inputs"
    first = canonical_json(freeze_manifest(first_inputs)) + "\n"
    second = canonical_json(freeze_manifest(make_real_inputs())) + "\n"
    assert first.encode() == second.encode()


def test_single_model_input_compiles_callback_with_fresh_sources() -> None:
    def emit(builder: TimelineBuilder, center: float) -> None:
        builder.emit(
            Pulse(
                "drive",
                PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=center,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=0.1),
                ),
            )
        )

    item = single_model_input(
        "custom_single",
        emit,
        {"pulse_sequence": "constant drive, then terminal readout"},
        ("dyson",),
    )
    with (
        closing((first := item.build()).source),
        closing((second := item.build()).source),
    ):
        assert first.source is not second.source
        assert first.name == second.name == "custom_single"
        assert first.families == second.families == ("dyson",)
        assert len(first.batch.points) == 1
        assert len(first.batch.tracks) == 1
        model = first.source.model((0,))
        assert model.energies_ghz.shape == (21,)
        np.testing.assert_allclose(
            first.batch.tracks[0].frequency_ghz,
            model.energies_ghz[1] - model.energies_ghz[0],
        )
        np.testing.assert_allclose(
            first.source.initial_density, second.source.initial_density
        )


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

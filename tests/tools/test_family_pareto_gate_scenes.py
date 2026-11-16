"""Numeric gate construction, source ownership and frozen handoff contracts."""

from contextlib import closing
from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import (
    canonical_json,
    read_physical_input,
    segment_spans,
)

from tests.simulate.evolution.contract.accuracy_cases import joint_case
from tools.family_pareto_gate_scenes import make_gate_inputs, numeric_gate_sequence
from tools.family_pareto_inputs import freeze_records


def test_gate_input_detaches_sources_and_freezes_numeric_provenance() -> None:
    (item,) = make_gate_inputs("multitone_gate_gaussian_n1")
    first, second = item.build(), item.build()
    first.source.close()
    with closing(second.source):
        model = second.source.model((0, 0, 0))
        (record,) = freeze_records([item])
        canonical_json(record.metadata)
        frozen = read_physical_input(record.physical)
        with closing(frozen.source):
            np.testing.assert_array_equal(
                frozen.source.model((0, 0, 0)).energies_ghz, model.energies_ghz
            )
            np.testing.assert_array_equal(
                frozen.source.initial_density, second.source.initial_density
            )
            assert segment_spans(frozen.batch) == segment_spans(second.batch)


def test_numeric_gate_sequence_preserves_simultaneity_and_absolute_carrier_phase() -> (
    None
):
    case = joint_case(constant=False)
    with closing(case.source):
        first, second = (
            replace(
                track,
                start_ns=np.array([0.0]),
                duration_ns=np.array([100.0]),
                frequency_ghz=track.frequency_ghz[:1],
                phase_rad=np.array([phase]),
                amplitude=track.amplitude[:1],
                envelope_index=track.envelope_index[:1],
                active=track.active[:1],
            )
            for track, phase in zip(case.batch.tracks, (0.2, -0.4), strict=True)
        )
        cz = tuple(
            replace(
                track,
                duration_ns=np.array([200.0]),
                frequency_ghz=np.array([0.73]),
                phase_rad=np.array([0.0]),
            )
            for track in (first, second)
        )
        batch = numeric_gate_sequence(
            case.source,
            {"pair": (first, second), "tail": (second,), "cz": cz},
            ("pair", "tail", "cz", "pair"),
            level=(0,),
        )
        assert len(batch.tracks) == 7
        assert [track.port for track in batch.tracks] == [
            "a",
            "b",
            "b",
            "a",
            "b",
            "a",
            "b",
        ]
        np.testing.assert_array_equal(
            [track.start_ns[0] for track in batch.tracks],
            [0, 0, 100, 200, 200, 400, 400],
        )
        np.testing.assert_array_equal(
            [track.duration_ns[0] for track in batch.tracks],
            [100, 100, 100, 200, 200, 100, 100],
        )
        np.testing.assert_array_equal(batch.end_ns, [500])
        # Absolute t=0 phases must not cancel the carrier's accumulated phase
        # when the same gate is repeated at a new start.
        for actual, template in zip(
            batch.tracks, (first, second, second, *cz, first, second), strict=True
        ):
            np.testing.assert_array_equal(actual.frequency_ghz, template.frequency_ghz)
            np.testing.assert_array_equal(actual.phase_rad, template.phase_rad)
            np.testing.assert_array_equal(actual.amplitude, template.amplitude)
            np.testing.assert_array_equal(
                actual.envelopes[0].values(np.array([0.0, 0.75, 1.5])),
                template.envelopes[0].values(np.array([0.0, 0.75, 1.5])),
            )
        assert segment_spans(batch) == (
            ((0.0, 100.0), (100.0, 200.0), (200.0, 400.0), (400.0, 500.0)),
        )

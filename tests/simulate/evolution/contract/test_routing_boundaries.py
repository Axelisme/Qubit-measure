"""Engine routing boundaries separate dispatch costs from numerical admission.

Checked full-drive gates can be pinned below their cold-cost break-even.
Sampled multitone RWA remains unavailable regardless of the price split.
"""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    EnvelopeSpec,
    NoValidKernel,
)

from .accuracy_cases import (
    AccuracyCase,
    accuracy_case,
    coherent_source,
    drive_track,
    linear_drive_model,
    timeline,
)
from .reference import converged_reference, density_error


def _joint_gate_case(count: int, duration_ns: float = 1.5) -> AccuracyCase:
    case = accuracy_case("joint_gate_inventory")
    batch = timeline(count, duration_ns)
    tracks = tuple(
        drive_track(
            batch,
            replace(
                track.envelopes[0],
                sample_ns=track.envelopes[0].sample_ns * (duration_ns / 1.5),
            ),
            frequency=float(track.frequency_ghz[0]),
            gain=float(track.amplitude[0]),
            port=track.port,
        )
        for track in case.batch.tracks
    )
    return replace(case, batch=replace(batch, tracks=tracks))


def test_whitelisted_multitone_gate_keeps_numerical_fallback() -> None:
    """A sole eligible gate executes without reviving sampled RWA."""
    count = 1
    case = _joint_gate_case(count)
    reference = converged_reference(case.source, case.batch)
    with closing(Engine(1, families=("multitone_rwa", "multitone_gate"))) as engine:
        result = engine.evolve(case.source, (case.batch,))
    assert result.paths.segments_by_kernel == {"multitone_gate": count}
    assert density_error(result.densities, reference.density) <= 2e-3
    # The default route's shared-frame premise needs its original 20 points;
    # it is independent of the one-use allowlist fallback above.
    shared = _joint_gate_case(20)
    shared_reference = converged_reference(shared.source, shared.batch)
    with closing(Engine(1)) as engine:
        for _ in range(2):
            result = engine.evolve(shared.source, (shared.batch,))
            assert result.paths.segments_by_kernel == {"common_frame": 20}
            assert density_error(result.densities, shared_reference.density) <= 2e-3


def test_pinned_multitone_gate_below_cost_break_even() -> None:
    """Pinning bypasses price, not validation; sampled RWA still rejects."""
    case = _joint_gate_case(1)
    with closing(Engine(1)) as engine:
        engine.evolve(case.source, (case.batch,))
    reference = converged_reference(case.source, case.batch)
    with closing(Engine(1, families=("multitone_gate",))) as engine:
        result = engine.evolve(case.source, (case.batch,))
    assert result.paths.segments_by_kernel == {"multitone_gate": 1}
    assert density_error(result.densities, reference.density) <= 2e-3
    with (
        closing(Engine(1, families=("multitone_rwa",))) as engine,
        pytest.raises(NoValidKernel) as rejected,
    ):
        engine.evolve(case.source, (case.batch,))
    assert rejected.value.domain == "multiple"
    assert rejected.value.attempted == ("multitone_rwa",)


# These inventories check losses far below cold break-even. Tests observe
# dispatch rather than pricing the key.
@pytest.mark.parametrize("count", [1, 10, 20])
def test_short_inventory_skips_unprofitable_gate_before_guard(count: int) -> None:
    """Twenty uses cannot repay a full-drive phase build."""
    case = _joint_gate_case(count)
    with closing(Engine(1, families=("multitone_gate", "dop853"))) as engine:
        query = engine.inspect_dispatch(case.source, (case.batch,))
    assert len(query.rows) == count
    for row in query.rows:
        assert row.family == "dop853"
        assert {c.family: c.guard for c in row.candidates} == {
            "dop853": "accepted",
            "multitone_gate": "untested_after_short_circuit",
        }


# At this pulse length enough reuse can beat the direct price even after
# including playback. Keep the 1.5 ns geometry for the common-frame contract.
_PRICE_CASE_DURATION_NS = 6.0


@pytest.mark.parametrize("prior_gate_use", [False, True], ids=["cold", "warm"])
def test_short_reuse_loses_to_dop853_without_building_gate(
    prior_gate_use: bool,
) -> None:
    """Short reuse does not repay a cold build, even after prior gate execution."""
    # This key used to win at 64 uses when the direct competitor was overpriced.
    count = 64
    case = _joint_gate_case(count, _PRICE_CASE_DURATION_NS)
    with (
        closing(case.source) as source,
        closing(Engine(1, families=("multitone_gate", "dop853"))) as engine,
    ):
        if prior_gate_use:
            long = _joint_gate_case(512, _PRICE_CASE_DURATION_NS).batch
            previous = engine.evolve(source, (long,))
            assert previous.paths.segments_by_kernel == {"multitone_gate": 512}
        query = engine.inspect_dispatch(source, (case.batch,))
        result = engine.evolve(source, (case.batch,), telemetry=True)
    assert len(query.rows) == count
    for row in query.rows:
        assert row.family == "dop853"
        assert {c.family: c.guard for c in row.candidates} == {
            "dop853": "accepted",
            "multitone_gate": "untested_after_short_circuit",
        }
    assert result.paths.segments_by_kernel == {"dop853": count}
    gate = result.paths.routing.get("multitone_gate")
    assert gate is None or gate.prepare_cpu_s == 0


@pytest.mark.parametrize("shared", [False, True])
def test_profitable_gate_is_offered_using_complete_inventory(shared: bool) -> None:
    """Local builders defer; a shared key runs its numerical guard in the caller."""
    count = 512
    duration = _PRICE_CASE_DURATION_NS
    case = _joint_gate_case(count if shared else 1, duration)
    work = case.batch
    if not shared:
        work = replace(
            work,
            end_ns=np.array([count * duration]),
            tracks=tuple(
                replace(track, start_ns=np.array([index * duration]))
                for index in range(count)
                for track in work.tracks
            ),
        )
    with closing(Engine(1, families=("multitone_gate", "dop853"))) as engine:
        query = engine.inspect_dispatch(case.source, (work,))
    assert len(query.rows) == count
    for row in query.rows:
        guards = {candidate.family: candidate.guard for candidate in row.candidates}
        if shared:
            assert guards["multitone_gate"] in ("accepted", "rejected")
        else:
            assert row.status == "deferred"
            assert row.family is None
            assert guards["multitone_gate"] == "deferred"


def test_profitable_gate_reuse_pays_only_its_own_build() -> None:
    """Frequent reuse can repay one cold build, but not a distinct key's build."""
    count = 512
    case = _joint_gate_case(count, _PRICE_CASE_DURATION_NS)
    rare = _joint_gate_case(1, _PRICE_CASE_DURATION_NS).batch
    rare = replace(
        rare,
        points=np.array([count]),
        tracks=tuple(
            replace(track, amplitude=track.amplitude * 1.1) for track in rare.tracks
        ),
    )
    with closing(Engine(1, families=("multitone_gate", "dop853"))) as engine:
        result = engine.evolve(case.source, (case.batch, rare), telemetry=True)
        query = engine.inspect_dispatch(case.source, (case.batch, rare))
    assert result.paths.segments_by_kernel == {"multitone_gate": count, "dop853": 1}
    assert result.paths.routing["multitone_gate"].rejected_rows == 0
    rare_rows = [row for row in query.rows if row.key.batch == 1]
    assert len(rare_rows) == 1
    assert rare_rows[0].family == "dop853"
    assert {c.family: c.guard for c in rare_rows[0].candidates} == {
        "dop853": "accepted",
        "multitone_gate": "untested_after_short_circuit",
    }


def _carrier_case(frequency: float) -> AccuracyCase:
    # Near-resonant strength excludes weak-drive Dyson above the boundary.
    model = replace(linear_drive_model(), energies_ghz=np.array([0.0, 0.05]))
    batch = timeline(2, 1.0)
    batch = replace(
        batch,
        tracks=(
            drive_track(batch, EnvelopeSpec("constant"), frequency=frequency, gain=20),
        ),
    )
    return AccuracyCase("carrier_boundary", coherent_source(model), batch, ())


@pytest.mark.parametrize(
    "frequency,family",
    [
        (sign * carrier, family)
        for sign in (1, -1)
        for carrier, family in (
            (0.029999, "slow_envelope"),
            (0.03, "floquet_period"),
            (0.030001, "floquet_period"),
        )
    ],
)
def test_default_absolute_carrier_boundary(frequency: float, family: str) -> None:
    case = _carrier_case(frequency)
    reference = converged_reference(case.source, case.batch)
    with closing(Engine(1)) as engine:
        result = engine.evolve(case.source, (case.batch,))
    assert dict(result.paths.segments_by_kernel) == {family: 2}
    assert density_error(result.densities, reference.density) <= 2e-3


def test_pinned_slow_envelope_rejects_exact_carrier_boundary() -> None:
    for frequency in (0.03, -0.03):
        case = _carrier_case(frequency)
        with (
            closing(Engine(1, families=("slow_envelope",))) as engine,
            pytest.raises(RuntimeError, match=r"domain plateau: points \[0, 1\]"),
        ):
            engine.evolve(case.source, (case.batch,))

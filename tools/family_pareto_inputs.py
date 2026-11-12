"""Frozen synthetic Pareto inputs; no lab configuration or pytest fixtures.

The physical inventory is shared with the accuracy table. Workloads lift each
representative to 96 points: three ordered gains, 32 complete repetitions each,
matching the public acquire benchmark's 3 x 32 scale (not acquire semantics).
"""

import hashlib
import json
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from functools import partial

import numpy as np
from zcu_tools.simulate.evolution import FluxSchedule, FluxWaveform, RoutingHistory

from tests.simulate.evolution.contract.accuracy_cases import (
    AccuracyCase,
    accuracy_case,
    history_case,
)
from tests.simulate.evolution.contract.guard_edge_cases import (
    EDGE_FAMILIES,
    guard_edge_case,
)
from tests.simulate.evolution.contract.guard_sweep_cases import (
    GATE_SAMPLES,
    SCALED_SAMPLES,
    gate_shape_bridge_case,
    gate_split_case,
    scaled_edge_case,
)

CASE_NAMES = (
    "idle_coherence",
    "weak_plateau",
    "strong_plateau",
    "shaped_complex",
    "two_carriers",
    "continuous_flux",
    "joint_gate_inventory",
    "low_carrier",
    "linear_shaped",
    "linear_plateau",
    "nearby_carriers",
    "adiabatic_ramp",
)


@dataclass(frozen=True)
class Input:
    """Stable name, fresh deterministic builder and why this complete input exists."""

    name: str
    build: Callable[[], AccuracyCase]
    reason: str


def multisegment_case() -> AccuracyCase:
    """Build the accuracy table's complete idle/drive/idle sequence."""
    case = accuracy_case("weak_plateau")
    pulse = replace(
        case.batch.tracks[0],
        start_ns=np.full(2, 0.3),
        duration_ns=np.full(2, 0.9),
        amplitude=np.full(2, 0.05),
    )
    return replace(
        case,
        name="multisegment/final",
        batch=replace(case.batch, tracks=(pulse,), segment_boundaries=((0.7,), (0.7,))),
    )


def representative(name: str) -> AccuracyCase:
    """Lift a representative's complete point timelines to 3 gains x 32 reps.

    Original durations, phases, envelopes, models and ragged flux spans survive.
    Gain factors are 1, 1.5, 2 (the acquire benchmark's .08, .12, .16 ratios).
    Every gain receives 32 full points, retaining multitone gate inventory.
    These are Engine density workloads, without ADC, readout or shot sampling.
    """
    case = accuracy_case(name)
    batch = case.batch
    index = np.tile(np.arange(32) % len(batch.points), 3)
    gains = np.repeat([1.0, 1.5, 2.0], 32)
    tracks = tuple(
        replace(
            t,
            start_ns=t.start_ns[index],
            duration_ns=t.duration_ns[index],
            frequency_ghz=t.frequency_ghz[index],
            phase_rad=t.phase_rad[index],
            amplitude=t.amplitude[index] * gains,
            envelope_index=t.envelope_index[index],
            active=t.active[index],
        )
        for t in batch.tracks
    )
    spans = [
        np.arange(int(batch.flux.offsets[int(i)]), int(batch.flux.offsets[int(i) + 1]))
        for i in index
    ]
    rows = np.concatenate(spans)
    flux = FluxSchedule(
        np.r_[0, np.cumsum([len(s) for s in spans])],
        batch.flux.start_ns[rows],
        batch.flux.level[rows],
    )
    waveform = None
    if batch.flux_waveform is not None:
        w = batch.flux_waveform
        spans = [
            np.arange(int(w.offsets[int(i)]), int(w.offsets[int(i) + 1])) for i in index
        ]
        rows = np.concatenate(spans)
        waveform = FluxWaveform(
            np.r_[0, np.cumsum([len(s) for s in spans])], w.time_ns[rows], w.flux[rows]
        )
    return replace(
        case,
        name="workload/" + name,
        batch=replace(
            batch,
            points=np.arange(96),
            tracks=tracks,
            flux=flux,
            end_ns=batch.end_ns[index],
            flux_waveform=waveform,
        ),
    )


def make_inputs() -> list[Input]:
    """Return the version-1 ordered accuracy, edge, sweep and workload inventory.

    Fallback tests reuse the unity gate and long split physical inputs already
    present here. Every input is tried with every family, so those fallback
    routes do not need duplicate oracles. History's two/eight-point inputs and
    the complete multisegment final-density input are included explicitly.
    """
    inputs = [
        Input(name, partial(accuracy_case, name), "original accuracy representative")
        for name in CASE_NAMES
    ]
    inputs += [
        Input(
            "guard_edge/" + f,
            partial(guard_edge_case, f),
            "guard edge inside existing numerical admission",
        )
        for f in EDGE_FAMILIES
    ]
    for family, scale, _ in SCALED_SAMPLES:
        build = partial(scaled_edge_case, family, scale)
        case = build()
        case.source.close()
        inputs.append(
            Input(case.name, build, "accuracy gain sweep, including crossing brackets")
        )
    for gamma, v, split, _ in GATE_SAMPLES:
        build_gate = partial(gate_split_case, gamma=gamma, v=v, split=split)
        case = build_gate()
        case.source.close()
        inputs.append(
            Input(
                case.name,
                build_gate,
                "accuracy split/bath sweep; long input also tests shaped_sambe fallback",
            )
        )
    for shaped in (False, True):
        build_bridge = partial(gate_shape_bridge_case, shaped=shaped)
        case = build_bridge()
        case.source.close()
        inputs.append(
            Input(
                case.name,
                build_bridge,
                "unity/nonunity shape bridge; unity also tests single_tone_gate fallback",
            )
        )
    inputs.append(
        Input(
            "multisegment/final", multisegment_case, "complete sequence final density"
        )
    )
    for count in (2, 8):
        inputs.append(
            Input(
                f"history/{count}",
                lambda count=count: history_case(count, RoutingHistory()),
                "history accuracy physical input; modes measured independently",
            )
        )
    inputs += [
        Input(
            "workload/" + name,
            partial(representative, name),
            "96 complete points = 3 gain factors [1,1.5,2] x 32 reps; "
            "same 3x32 scale as acquire benchmark; "
            f"represents {name} domain and its original families",
        )
        for name in CASE_NAMES
    ]
    return inputs


def canonical_json(value: object) -> str:
    """Encode deterministic numeric records, with complex numbers as real/imag pairs."""

    def encode(item: object) -> object:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, complex):
            return {"real": item.real, "imag": item.imag}
        raise TypeError(f"unsupported manifest value: {type(item).__name__}")

    return json.dumps(
        value, default=encode, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: object) -> str:
    """SHA-256 of canonical_json(value), including every numeric input entry."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def physical_input(
    case: AccuracyCase, *, include_identity: bool = True
) -> dict[str, object]:
    """Serialize complete timeline, initial density, models and optional expansion.

    Builders in this inventory use level-independent model replacements; all
    scheduled levels and the DC model are listed. Continuous Hamiltonians are
    fully specified by the supplied expansion. No mutable history is serialized.
    include_identity=False drops only the nonphysical model namespace for oracle
    sharing; complete manifests retain that namespace for route reconstruction.
    """
    levels = sorted(
        {tuple(int(v) for v in row) for row in case.batch.flux.level} | {(0,)}
    )
    models = [asdict(case.source.model(level)) for level in levels]
    if not include_identity:
        for model in models:
            model.pop("identity")
    return {
        "batch": asdict(case.batch),
        "initial_density": case.source.initial_density,
        "models": models,
        "expansion": asdict(case.source.flux_expansion())
        if case.batch.flux_waveform is not None
        else None,
        "slow_envelope_segments_per_period": case.source.slow_envelope_segments_per_period,
    }


def freeze_manifest(inputs: list[Input]) -> dict[str, object]:
    """Build all inputs before measuring; preserve complete values, hashes and reasons."""
    records = []
    for item in inputs:
        case = item.build()
        with closing(case.source):
            physical = physical_input(case)
            records.append(
                {
                    "name": item.name,
                    "reason": item.reason,
                    "expected_families": case.families,
                    "physical_sha256": digest(physical),
                    "oracle_sha256": digest(
                        physical_input(case, include_identity=False)
                    ),
                    "physical": physical,
                }
            )
    return {"version": 1, "inputs": records}

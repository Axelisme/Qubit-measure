"""Producer-owned scene declarations and frozen numeric manifest metadata.

Accuracy/guard matrices stay in tests. The retained synthetic idle scene is
explicitly declared here, independently of those mutable test inventories.
"""

from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter
from zcu_tools.simulate.evolution import (
    FluxSchedule,
    FrozenRecord,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
    read_physical_input,
    read_scene_records,
)
from zcu_tools.simulate.evolution import (
    canonical_json as canonical_json,
)
from zcu_tools.simulate.evolution import physical_input as numeric_input

# Cache-only consumers share labels without importing model/program builders.
# ADR-0072 keeps those producers out of the measurement import graph.
MULTITONE_PRICE_SCENES = (
    "mtrwa_c2b_l16_128",
    "mtrwa_c2b_l20_128",
    "mtrwa_c2b_l16_512",
    "mtrwa_star_base",
    "mtrwa_star_action_low",
    "mtrwa_star_action_high",
    "mtrwa_star_dimension_low",
    "mtrwa_star_dimension_high",
    "mtrwa_star_bandwidth_low",
    "mtrwa_star_bandwidth_high",
    "mtrwa_star_duration_low",
    "mtrwa_star_duration_high",
    "mtrwa_star_n20",
    "mtrwa_star_n100",
)


@dataclass(frozen=True)
class Scene:
    """One named Engine workload with a caller-owned source.

    name is the stable scene label. source owns numeric model/history state
    and must be closed after use. batch contains complete ordered points.
    families lists independently pinnable targets, not per-segment preferences.
    """

    name: str
    source: ModelSource
    batch: TimelineBatch
    families: tuple[str, ...]


@dataclass(frozen=True)
class Input:
    """Stable producer declaration; build returns a fresh owned Scene each time.

    name is the inventory label and reason explains representation. companions
    names additional pinned families. expected_routes maps targets to family ->
    (logical row count, cumulative ns). metadata/manifest carry caller provenance.
    category separates synthetic and real frozen manifests; solver selection
    belongs to the reference producer's caller.
    companions_by_target replaces companions for listed targets, including an
    empty tuple to pin a target alone. Empty declarations retain the original
    single-family synthetic protocol. None of these fields changes physics.
    """

    name: str
    build: Callable[[], Scene]
    reason: str
    companions: tuple[str, ...] = ()
    expected_routes: dict[str, dict[str, tuple[int, float]]] = field(
        default_factory=dict
    )
    metadata: dict[str, object] = field(default_factory=dict)
    manifest: dict[str, object] = field(default_factory=dict)
    category: str = "synthetic"
    companions_by_target: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def companions_for(self, target: str) -> tuple[str, ...]:
        """Return the target override, or shared companions when absent."""
        return self.companions_by_target.get(target, self.companions)


def _read_scene(record: FrozenRecord, families: tuple[str, ...]) -> Scene:
    frozen = read_physical_input(record.physical)
    return Scene(record.name, frozen.source, frozen.batch, families)


def read_inputs(indexes: Sequence[Path], scene_cache: Path) -> list[Input]:
    """Read ordered frozen records and decode the tools-owned scene declarations.

    indexes supplies versioned identity lists in caller order; scene_cache is an
    explicit numeric cache directory. Cross-index duplicate names fail. Each
    returned Input keeps its record name, reason, targets, companions, routes
    and provenance. No source is built while reading the declarations. Every
    Input.build() reconstructs a fresh Scene through read_physical_input; the
    caller must close its source. Sources do not share routing history.
    File/container errors, missing declarations and invalid metadata propagate;
    no producer, solver or legacy reader is used.
    """
    from tools.family_pareto_storage import indexed_scenes, missing_scene_command

    entries = indexed_scenes(indexes)
    try:
        records = read_scene_records(scene_cache, (entry for _, entry in entries))
    except FileNotFoundError as error:
        raise missing_scene_command(error, scene_cache) from error
    return [input_from_record(record) for record in records]


def input_from_record(record: FrozenRecord) -> Input:
    """Decode one codec record's tools metadata, without building a source.

    Return an Input whose build creates a fresh caller-owned Scene and history.
    Missing or invalid tools declarations fail through their typed decoder.
    The record's numeric content has already been validated by its codec owner.
    """
    metadata = TypeAdapter(dict[str, object]).validate_python(record.metadata)
    families = TypeAdapter(tuple[str, ...]).validate_python(
        metadata.pop("expected_families")
    )
    return TypeAdapter(Input).validate_python(
        {
            **metadata,
            "name": record.name,
            "build": partial(_read_scene, record, families),
        }
    )


def idle_coherence_scene() -> Scene:
    """Build the original three-level, two-point 1.5 ns synthetic idle input.

    Keep the historical energies, bath, quadratures and coherent initial state
    unchanged, so the numeric hash and reference key remain stable. No solver,
    environment, frontend or test inventory is imported. Caller closes source.
    """
    rates = np.zeros((3, 3))
    rates[0, 1], rates[1, 0], rates[1, 2] = 0.015, 0.002, 0.007
    slopes = np.arange(3) * 0.03
    x = np.diag(np.ones(2), 1) + np.diag(np.ones(2), -1)
    y = 1j * (np.diag(np.ones(2), 1) - np.diag(np.ones(2), -1))
    model = SecularModel(
        (0,),
        "synthetic-3",
        np.arange(3, dtype=float) * 0.13,
        np.eye(3, dtype=complex),
        (SecularDrive("drive", x.astype(np.complex128), y.astype(np.complex128)),),
        rates,
        (slopes[:, None] - slopes[None, :]) ** 2 / 2,
    )
    vector = np.exp(0.7j * np.arange(3))
    vector /= np.linalg.norm(vector)
    source = ModelSource(
        np.outer(vector, vector.conj()), lambda level: replace(model, level=level)
    )
    batch = TimelineBatch(
        np.arange(2),
        (),
        FluxSchedule(np.arange(3), np.zeros(2), np.zeros((2, 1), dtype=np.int64)),
        np.full(2, 1.5),
    )
    return Scene("idle_coherence", source, batch, ("t4",))


def make_inputs() -> list[Input]:
    """Return the explicitly declared synthetic producer inventory.

    Guard grids and history fixtures belong to tests-side contracts, not the
    measurement scene list. Real complete sequences are declared separately.
    """
    return [
        Input(
            "idle_coherence",
            idle_coherence_scene,
            "three-level idle with complex coherence",
        )
    ]


def physical_input(case: Scene, *, include_identity: bool = True) -> dict[str, object]:
    """Snapshot a Scene through the simulator-owned numeric codec without solving."""
    return numeric_input(case.source, case.batch, include_identity=include_identity)


def _scene_metadata(item: Input, scene: Scene) -> dict[str, object]:
    return {
        "reason": item.reason,
        "expected_families": scene.families,
        **(
            {"companions_by_target": item.companions_by_target}
            if item.companions_by_target
            else {}
        ),
        **(
            {
                "category": item.category,
                "metadata": item.metadata,
                "manifest": item.manifest,
                "expected_routes": item.expected_routes,
                "companions": item.companions,
            }
            if item.category == "real"
            else {}
        ),
    }


def freeze_records(inputs: list[Input]) -> tuple[FrozenRecord, ...]:
    """Snapshot ordered declarations and close sources, without evolving.

    Scene declarations are tools-owned opaque metadata. The simulator manifest
    writer calculates hashes and owns the container; no solver policy is encoded.
    Builder and serialization failures propagate. Source arrays stay in the
    returned numeric records after the sources are closed.
    """
    records = []
    for item in inputs:
        scene = item.build()
        with closing(scene.source):
            records.append(
                FrozenRecord(
                    item.name,
                    physical_input(scene),
                    _scene_metadata(item, scene),
                )
            )
    return tuple(records)

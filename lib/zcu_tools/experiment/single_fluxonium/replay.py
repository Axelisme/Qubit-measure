"""Author-only single-fluxonium replay, using the neutral resource archive."""

from importlib.metadata import version
from pathlib import Path

from zcu_tools.core.simulation import RoundRequest, RoundSignals
from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.resources.simulation_replay import read_replay
from zcu_tools.resources.single_fluxonium import SingleFluxoniumSettings
from zcu_tools.simulate.evolution import SEGMENT_NUMERIC_VERSION

BACKEND_VERSION = "single-fluxonium-segment-v1"


def replay_versions() -> tuple[tuple[str, str], ...]:
    """Return exact backend/model/numeric/RNG and dependency compatibility pairs.

    Values describe private reproducibility requirements, not public provenance.
    Replay checks all pairs before recomputing any physical model.
    Missing installed dependency metadata raises PackageNotFoundError.
    """
    return (
        ("backend", BACKEND_VERSION),
        ("model", "single-fluxonium-v1"),
        ("numeric", SEGMENT_NUMERIC_VERSION),
        ("rng", "PCG64/SeedSequence-v2"),
        *((name, version(name)) for name in ("numpy", "scipy", "scqubits")),
    )


def replay_round(path: Path) -> IntegratedRound | RoundSignals:
    """Recompute one archived round without live devices or current YAML/assets.

    path is an immutable round path returned by resources.save_replay. Requires
    single-fluxonium settings and exact backend/model/numeric/RNG/dependency
    versions. Wrong backend, model identity or numeric references raise ValueError.
    Storage integrity and solver failures propagate. No earlier round executes.
    Returns unnormalized statistics/deferred raw or unrounded decimated shots;
    these contain model truth and must not enter ordinary GUI/RPC metadata.
    """
    from .runtime import SingleFluxoniumSource

    inputs = read_replay(path)
    acquire, record = inputs.acquire, inputs.round
    if not isinstance(acquire.settings, SingleFluxoniumSettings):
        raise ValueError("single-fluxonium replay requires single-fluxonium settings")
    if acquire.versions != replay_versions():
        raise ValueError("private replay backend/model/RNG/numeric version mismatch")
    if record.numeric_manifest is not None or acquire.library_path is not None:
        raise ValueError("private replay does not accept numeric record references")
    source = SingleFluxoniumSource(acquire.settings, DeviceManager())
    try:
        if (
            record.snapshot.version != "single-fluxonium-v1"
            or record.snapshot.model_identity != source.identity
        ):
            raise ValueError("private replay model identity mismatch")
        return source.execute(
            RoundRequest(acquire.request, record.snapshot, record.index)
        )
    finally:
        source.close()

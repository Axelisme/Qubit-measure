"""Author-only FTF replay coordination; formats stay with their owning tools."""

from importlib.metadata import version
from pathlib import Path

from zcu_tools.core.simulation import RoundRequest, RoundSignals, RoundStatistics
from zcu_tools.device import DeviceManager
from zcu_tools.resources.simulation_replay import read_replay
from zcu_tools.simulate.evolution import SEGMENT_NUMERIC_VERSION

BACKEND_VERSION = "ftf-segment-v6"


def replay_versions() -> tuple[tuple[str, str], ...]:
    """Exact FTF backend/model/numeric/RNG dependency versions required for replay.

    This private compatibility policy deliberately rejects version differences;
    equal seeds do not promise bit-exact results across library versions.
    """
    return (
        ("backend", BACKEND_VERSION),
        ("model", "ftf-v2"),
        ("numeric", SEGMENT_NUMERIC_VERSION),
        ("rng", "PCG64/SeedSequence-v2"),
        *((name, version(name)) for name in ("numpy", "scipy", "scqubits")),
    )


def replay_round(path: Path) -> RoundStatistics | RoundSignals:
    """Reload and execute only one saved private FTF round, without live devices.

    path is an immutable round JSON path from resources.simulation_replay.
    Resolves full captured settings, controls/envelope samples, sampling identity
    and round snapshot. Checks exact backend/model/RNG/numeric versions plus
    dependencies before recomputing. Numeric records and
    current YAML/assets are not consulted. No previous acquire or round runs.
    Returns compact statistics or unrounded traces, not QICK-normalized results.
    Version-incompatible inputs raise explicitly. This is an
    author diagnostic interface containing truth, not normal GUI/RPC metadata.
    """
    from .runtime import FTFSource

    inputs = read_replay(path)
    acquire, record = inputs.acquire, inputs.round
    if acquire.versions != replay_versions():
        raise ValueError("private replay backend/model/RNG/numeric version mismatch")
    if record.numeric_manifest is not None or acquire.library_path is not None:
        raise ValueError("private replay does not accept numeric record references")
    source = FTFSource(acquire.settings, DeviceManager())
    try:
        if (
            record.snapshot.version != "ftf-v2"
            or record.snapshot.model_identity != source.identity
        ):
            raise ValueError("private replay model identity mismatch")
        request = RoundRequest(acquire.request, record.snapshot, record.index)
        return source.execute(request)
    finally:
        source.close()

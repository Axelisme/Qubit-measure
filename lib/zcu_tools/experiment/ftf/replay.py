"""Author-only FTF replay coordination; formats stay with their owning tools."""

from importlib.metadata import version
from pathlib import Path

from zcu_tools.core.simulation import RoundRequest, RoundSignals
from zcu_tools.device import DeviceManager
from zcu_tools.resources.simulation_replay import read_replay
from zcu_tools.simulate.evolution import NUMERIC_VERSION, EvolutionLibrary


def replay_versions() -> tuple[tuple[str, str], ...]:
    """Exact FTF backend/model/numeric/RNG dependency versions required for replay.

    This private compatibility policy deliberately rejects version differences;
    equal seeds do not promise bit-exact results across library versions.
    """
    return (
        ("backend", "ftf-direct-v1"),
        ("model", "ftf-v1"),
        ("numeric", NUMERIC_VERSION),
        ("rng", "PCG64/SeedSequence-v1"),
        *((name, version(name)) for name in ("numpy", "scipy", "scqubits")),
    )


def replay_round(path: Path) -> RoundSignals:
    """Reload and execute only one saved private FTF round, without live devices.

    path is an immutable round JSON path from resources.simulation_replay.
    Resolves full captured settings, controls/envelope samples, sampling identity
    and round snapshot. Checks exact backend/model/RNG/numeric versions plus
    every referenced numeric artifact before executing. New publications and
    current YAML/assets are not consulted. No previous acquire or round runs.
    Returns unrounded float64 ADC sums/traces, not QICK-normalized results.
    Missing/corrupt/version-incompatible artifacts raise explicitly. This is an
    author diagnostic interface containing truth, not normal GUI/RPC metadata.
    """
    from .runtime import FTFSource

    inputs = read_replay(path)
    acquire, record = inputs.acquire, inputs.round
    if acquire.versions != replay_versions():
        raise ValueError("private replay backend/model/RNG/numeric version mismatch")
    library = (
        EvolutionLibrary(acquire.library_path)
        if acquire.library_path is not None
        else None
    )
    tables = None
    if record.numeric_manifest is not None:
        if library is None:
            raise ValueError("private replay numeric manifest has no library")
        tables = library.restore_replay(record.numeric_manifest)
    elif library is not None:
        raise ValueError("private replay missing numeric manifest")
    source = FTFSource(acquire.settings, DeviceManager(), library=library)
    if (
        record.snapshot.version != "ftf-v1"
        or record.snapshot.model_identity != source.identity
    ):
        raise ValueError("private replay model identity mismatch")
    request = RoundRequest(acquire.request, record.snapshot, record.index)
    return source.execute(request, tables=tables)

"""Single-fluxonium private compatibility identity, separate from FTF replay.

Input archiving and replay execution remain unavailable until single-cavity
acquisition is connected. No files are read or written by this module.
"""

from importlib.metadata import version

from zcu_tools.simulate.evolution import SEGMENT_NUMERIC_VERSION

BACKEND_VERSION = "single-fluxonium-segment-v1"


def replay_versions() -> tuple[tuple[str, str], ...]:
    """Return exact backend/model/numeric/RNG and dependency compatibility pairs.

    Values describe private reproducibility requirements, not public provenance.
    Matching versions alone do not enable acquisition or archive execution.
    Missing installed dependency metadata raises PackageNotFoundError.
    """
    return (
        ("backend", BACKEND_VERSION),
        ("model", "single-fluxonium-v1"),
        ("numeric", SEGMENT_NUMERIC_VERSION),
        ("rng", "PCG64/SeedSequence-v2"),
        *((name, version(name)) for name in ("numpy", "scipy", "scqubits")),
    )

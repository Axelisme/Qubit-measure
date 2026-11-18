"""Author-only FTF replay coordination; formats stay with their owning tools."""

from contextlib import closing
from dataclasses import dataclass
from importlib.metadata import version
from math import prod
from pathlib import Path

from qick import QickConfig

from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.simulate.boundary import RoundSignals
from zcu_tools.simulate.evolution import SEGMENT_NUMERIC_VERSION, Engine
from zcu_tools.simulators.ftf.config import FTFSettings
from zcu_tools.simulators.replay import read_replay

BACKEND_VERSION = "ftf-segment-v14"
_VERSION_MISMATCH = "private replay backend/model/RNG/numeric version mismatch"
_DIMENSION_FIELD = "selected_dimension"
_PILOT_PREFIX = "ftf-v14:pilot-point:"


@dataclass(frozen=True)
class ReplayShape:
    """The archived acquire dimension and the target round's solve partition.

    dimension is the selected positive state count, at most the upper basis.
    pilot_point is the C-order point solved alone at the upper basis, or None
    to solve the full inventory together at dimension.
    """

    dimension: int
    pilot_point: int | None


def pilot_manifest(point: int | None) -> str | None:
    """Encode a C-order pilot point in the round record; None writes JSON null."""
    return None if point is None else f"{_PILOT_PREFIX}{point}"


def replay_versions(
    *, execution_device: str, workers: int, selected_dimension: int | None = None
) -> tuple[tuple[str, str], ...]:
    """Return FTF versions and execution metadata for the private acquire.

    execution_device is the PathDiagnostics device identity for the saved
    round. workers is the positive configured host worker count; replay
    retains it because CPU/GPU pricing uses parallel CPU width. Equal seeds
    do not promise bit-exact results across library versions. Include the
    acquire's selected_dimension for a complete archive; None omits that field
    when comparing only versions and execution conditions.
    """
    versions = (
        ("backend", BACKEND_VERSION),
        ("model", "ftf-v2"),
        ("numeric", SEGMENT_NUMERIC_VERSION),
        ("rng", "PCG64/SeedSequence-v2"),
        *((name, version(name)) for name in ("numpy", "scipy", "scqubits")),
        ("execution_device", execution_device),
        ("workers", str(workers)),
    )
    if selected_dimension is None:
        return versions
    return (*versions, (_DIMENSION_FIELD, str(selected_dimension)))


def _recorded_execution(
    versions: tuple[tuple[str, str], ...], *, upper: int
) -> tuple[str, int, int]:
    fields = dict(versions)
    try:
        device, workers = fields["execution_device"], int(fields["workers"])
    except (KeyError, ValueError) as error:
        raise ValueError(_VERSION_MISMATCH) from error
    base = tuple(pair for pair in versions if pair[0] != _DIMENSION_FIELD)
    if workers < 1 or base != replay_versions(execution_device=device, workers=workers):
        raise ValueError(_VERSION_MISMATCH)
    dimensions = [value for name, value in versions if name == _DIMENSION_FIELD]
    if not dimensions:
        raise ValueError("private replay missing selected dimension")
    if len(dimensions) != 1:
        raise ValueError("private replay requires one selected dimension")
    try:
        dimension = int(dimensions[0])
    except ValueError as error:
        raise ValueError(
            "private replay selected dimension must be an integer"
        ) from error
    if dimension < 1 or str(dimension) != dimensions[0]:
        raise ValueError("private replay selected dimension must be a positive integer")
    if dimension > upper:
        raise ValueError(
            f"private replay selected dimension exceeds upper bound {upper}"
        )
    return device, workers, dimension


def _recorded_pilot_point(manifest: str | None, points: int) -> int | None:
    if manifest is None:
        return None
    if not manifest.startswith(_PILOT_PREFIX):
        raise ValueError("private replay does not accept numeric record references")
    code = manifest.removeprefix(_PILOT_PREFIX)
    try:
        point = int(code)
    except ValueError as error:
        raise ValueError("private replay pilot point must be an integer") from error
    if not 0 <= point < points:
        raise ValueError(
            f"private replay pilot point {point} out of range for {points} points"
        )
    if str(point) != code:
        raise ValueError("private replay pilot point must be a canonical integer")
    return point


def _require_execution_device(recorded: str, current: str) -> None:
    if current != recorded:
        raise ValueError(
            f"private replay execution device mismatch: recorded {recorded}, "
            f"current {current}"
        )


def replay_round(path: Path, *, soccfg: QickConfig) -> IntegratedRound | RoundSignals:
    """Reload and execute only one saved private FTF round, without live devices.

    path is an immutable round JSON path from resources.simulation_replay.
    soccfg must be the QICK profile used while recording. The caller supplies it;
    the archive does not contain it and replay does not verify profile equality.
    Resolves full captured settings, controls/envelope samples, sampling identity
    and round snapshot. Checks exact backend/model/RNG/numeric and dependency
    versions before recomputing. Uses the recorded worker count and state
    count. When the round has a pilot point, recomputes its isolated upper
    result without selecting a point, checking a drop, or verifying a smaller
    basis. A CPU record never checks for OpenCL. A GPU record uses automatic
    routing and must run on the recorded OpenCL identity. Incompatible versions,
    device identities, or invalid state counts or pilot points raise ValueError.
    No numeric records, current YAML/assets or previous acquires or rounds
    are used. Returns compact statistics or unrounded traces, not
    QICK-normalized results. This diagnostic contains private model inputs,
    not normal GUI/RPC metadata.
    """
    from zcu_tools.simulators.ftf.environment.runtime import FTFSource

    inputs = read_replay(path)
    acquire, record = inputs.acquire, inputs.round
    if not isinstance(acquire.settings, FTFSettings):
        raise ValueError("FTF replay requires FTF settings")
    device, workers, dimension = _recorded_execution(
        acquire.versions, upper=acquire.settings.numerics.truncation.retained_levels
    )
    if acquire.library_path is not None:
        raise ValueError("private replay does not accept numeric record references")
    point = _recorded_pilot_point(
        record.numeric_manifest, prod(acquire.request.point_dims)
    )
    shape = ReplayShape(dimension, point)
    backend = "cpu" if device == "cpu" else "auto"
    with (
        closing(Engine(workers, backend=backend)) as engine,
        closing(
            FTFSource(acquire.settings, DeviceManager(), soccfg=soccfg, engine=engine)
        ) as source,
    ):
        _require_execution_device(device, engine.detect_execution_device())
        if (
            record.snapshot.version != "ftf-v2"
            or record.snapshot.model_identity != source.identity
        ):
            raise ValueError("private replay model identity mismatch")
        session = source.open_acquire(acquire.request, replay=shape)
        result = session.sample_round(record.snapshot, record.index)
        # Device detection can succeed while the needed dimension fails to
        # compile. Don't return CPU results for an archive recorded on GPU.
        _require_execution_device(device, source.evaluations[-1].paths.execution_device)
        return result

"""Author-only FTF replay coordination; formats stay with their owning tools."""

from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from qick import QickConfig

from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.simulate.boundary import RoundRequest, RoundSignals
from zcu_tools.simulate.evolution import SEGMENT_NUMERIC_VERSION, Engine
from zcu_tools.simulators.ftf.config import FTFSettings
from zcu_tools.simulators.replay import read_replay

BACKEND_VERSION = "ftf-segment-v13"
_VERSION_MISMATCH = "private replay backend/model/RNG/numeric version mismatch"


def replay_versions(
    *, execution_device: str, workers: int
) -> tuple[tuple[str, str], ...]:
    """Return strict FTF version pairs and the engine execution conditions.

    execution_device is the PathDiagnostics device identity for the saved
    round. workers is the positive configured host worker count; replay
    retains it because CPU/GPU pricing uses parallel CPU width. Equal seeds
    do not promise bit-exact results across library versions.
    """
    return (
        ("backend", BACKEND_VERSION),
        ("model", "ftf-v2"),
        ("numeric", SEGMENT_NUMERIC_VERSION),
        ("rng", "PCG64/SeedSequence-v2"),
        *((name, version(name)) for name in ("numpy", "scipy", "scqubits")),
        ("execution_device", execution_device),
        ("workers", str(workers)),
    )


def _recorded_execution(versions: tuple[tuple[str, str], ...]) -> tuple[str, int]:
    fields = dict(versions)
    try:
        device, workers = fields["execution_device"], int(fields["workers"])
    except (KeyError, ValueError) as error:
        raise ValueError(_VERSION_MISMATCH) from error
    if workers < 1 or versions != replay_versions(
        execution_device=device, workers=workers
    ):
        raise ValueError(_VERSION_MISMATCH)
    return device, workers


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
    versions before recomputing. Uses the recorded Engine worker count. A CPU
    record never checks for OpenCL; a GPU record uses automatic routing
    and must detect and run on the recorded OpenCL identity. Incompatible
    versions or device identities raise ValueError.
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
    device, workers = _recorded_execution(acquire.versions)
    if record.numeric_manifest is not None or acquire.library_path is not None:
        raise ValueError("private replay does not accept numeric record references")
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
        request = RoundRequest(acquire.request, record.snapshot, record.index)
        result = source.execute(request)
        # Device detection can succeed while the needed dimension fails to
        # compile. Don't return CPU results for an archive recorded on GPU.
        _require_execution_device(device, source.evaluations[-1].paths.execution_device)
        return result

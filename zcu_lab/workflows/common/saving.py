"""Save workflow acquisitions using core records and framework operation facts."""

from pathlib import Path

from zcu_tools.experiment.axes_spec import AxesSpec, GroupedAxesSpec
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.native_persistence import save_run
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.workflows import Completed, RunIdentity, record_with_metadata

from zcu_lab.v2.twotone.freq.core import FreqCfg, FreqExp, FreqResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiCfg, LenRabiExp, LenRabiResult
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import (
    FreqGainCfg,
    FreqGainExp,
    FreqGainResult,
)
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Exp, T1Result
from zcu_lab.v2.twotone.time_domain.t2echo.core import (
    T2EchoCfg,
    T2EchoExp,
    T2EchoResult,
)


def save_native[Cfg: ExpCfgModel, Result](
    source: Completed[RunRecord[Cfg, Result]],
    destination: Path,
    *,
    spec: AxesSpec[Result, Cfg] | GroupedAxesSpec[Result, Cfg],
    snapshot: RunSnapshot,
    identity: RunIdentity,
) -> None:
    """Save a completed core record with its fixed workflow acquisition facts.

    source.value supplies cfg/result; source.facts supplies UTC boundaries,
    iteration, attempt, stopped status and the pre-operation device snapshot.
    identity is env.identity for that workflow run; snapshot is the caller's
    historical entry/point capture. spec supplies the experiment tag and native
    array mapping. Compose existing RunMetadata without capturing new facts.
    destination is an exact new filename whose parent exists. Native schema,
    metadata, collision and I/O errors propagate; no overwrite or retry occurs.
    """
    record = record_with_metadata(
        source, identity=identity, experiment_tag=spec.tag, snapshot=snapshot
    )
    save_run(record, destination, spec=spec)


def save_frequency(
    source: Completed[RunRecord[FreqCfg, FreqResult]],
    destination: Path,
    *,
    snapshot: RunSnapshot,
    identity: RunIdentity,
) -> None:
    """Preserve spectroscopy IQ and its rounded MHz coordinate.

    source is env.run's completed core record and operation facts. Pass the same
    run's env.identity and historical snapshot. destination is an exact new
    native filename; its parent must exist. Schema, collision and I/O errors
    propagate without recapture or retry.
    """
    save_native(
        source,
        destination,
        spec=FreqExp().persistence_spec(),
        snapshot=snapshot,
        identity=identity,
    )


def save_rabi(
    source: Completed[RunRecord[LenRabiCfg, LenRabiResult]],
    destination: Path,
    *,
    snapshot: RunSnapshot,
    identity: RunIdentity,
) -> None:
    """Preserve Length-Rabi IQ and its rounded microsecond coordinate.

    source is env.run's completed core record and operation facts. Pass the same
    run's env.identity and historical snapshot. destination is an exact new
    native filename; its parent must exist. Schema, collision and I/O errors
    propagate without recapture or retry.
    """
    save_native(
        source,
        destination,
        spec=LenRabiExp().persistence_spec(),
        snapshot=snapshot,
        identity=identity,
    )


def save_readout(
    source: Completed[RunRecord[FreqGainCfg, FreqGainResult]],
    destination: Path,
    *,
    snapshot: RunSnapshot,
    identity: RunIdentity,
) -> None:
    """Preserve readout SNR with frequency and gain coordinates.

    source is env.run's completed core record and operation facts. Pass the same
    run's env.identity and historical snapshot. destination is an exact new
    native filename; its parent must exist. Schema, collision and I/O errors
    propagate without recapture or retry.
    """
    save_native(
        source,
        destination,
        spec=FreqGainExp().persistence_spec(),
        snapshot=snapshot,
        identity=identity,
    )


def save_t1(
    source: Completed[RunRecord[T1Cfg, T1Result]],
    destination: Path,
    *,
    snapshot: RunSnapshot,
    identity: RunIdentity,
) -> None:
    """Preserve relaxation IQ and its rounded microsecond delay coordinate.

    source is env.run's completed core record and operation facts. Pass the same
    run's env.identity and historical snapshot. destination is an exact new
    native filename; its parent must exist. Schema, collision and I/O errors
    propagate without recapture or retry.
    """
    save_native(
        source,
        destination,
        spec=T1Exp().persistence_spec(),
        snapshot=snapshot,
        identity=identity,
    )


def save_t2echo(
    source: Completed[RunRecord[T2EchoCfg, T2EchoResult]],
    destination: Path,
    *,
    snapshot: RunSnapshot,
    identity: RunIdentity,
) -> None:
    """Preserve Hahn-echo IQ and its rounded total-delay coordinate.

    source is env.run's completed core record and operation facts. Pass the same
    run's env.identity and historical snapshot. destination is an exact new
    native filename; its parent must exist. Schema, collision and I/O errors
    propagate without recapture or retry.
    """
    save_native(
        source,
        destination,
        spec=T2EchoExp().persistence_spec(),
        snapshot=snapshot,
        identity=identity,
    )

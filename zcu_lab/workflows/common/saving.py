"""common saving helpers for lab workflows."""

from pathlib import Path

from zcu_tools.datafile import RunMetadata, RunSnapshot
from zcu_tools.experiment.acquisition import AcquisitionEvidence
from zcu_tools.experiment.axes_spec import AxesSpec, GroupedAxesSpec
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.native_persistence import save_run
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.workflows import Completed

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
from zcu_lab.workflows.common.acquire import (
    FrequencyAcquisition,
    RabiAcquisition,
    ReadoutAcquisition,
    T1Acquisition,
    T2EchoAcquisition,
)


def save_native[Cfg: ExpCfgModel, Result](
    source: RunRecord[Cfg, Result],
    destination: Path,
    *,
    spec: AxesSpec[Result, Cfg] | GroupedAxesSpec[Result, Cfg],
    snapshot: RunSnapshot,
    evidence: AcquisitionEvidence,
) -> None:
    """Write a typed native record at the caller's exact destination.

    source contains the complete cfg/result pair; its cfg must match spec's
    cfg_type. destination names the native file, not a directory. spec declares
    the experiment tag, array mapping and SI/discrete units. snapshot supplies
    the host-captured historical entry/point state. evidence supplies acquisition
    identity, timestamps, completion, provenance and captured device state.
    Build RunMetadata from this fixed evidence and snapshot, then call save_run.
    Existing destinations raise FileExistsError. Invalid cfg/result/spec/metadata
    raises ValueError or TypeError; I/O errors propagate. No live capture or
    filename inference occurs.
    """
    metadata = RunMetadata(
        run_id=evidence.run_id,
        experiment=spec.tag,
        started_at=evidence.started_at.isoformat(),
        finished_at=evidence.finished_at.isoformat(),
        completion=evidence.completion,
        labber_path=None,
        snapshot=snapshot,
        provenance=evidence.provenance,
        device_snapshot=evidence.device_snapshot,
    )
    save_run(source, destination, spec=spec, metadata=metadata)


def save_frequency(
    source: Completed[FreqCfg, FrequencyAcquisition],
    destination: Path,
    *,
    snapshot: RunSnapshot,
) -> None:
    """Write native spectroscopy data to the exact, not-yet-existing path.

    source is Engine's complete cfg/result pair. snapshot is the host-captured
    historical entry state. Preserve the Result's acquisition evidence. Parent
    must exist; schema, existing-destination and I/O errors propagate.
    """
    record: RunRecord[FreqCfg, FreqResult] = RunRecord(source.cfg, source.result)
    save_native(
        record,
        destination,
        spec=FreqExp().persistence_spec(),
        snapshot=snapshot,
        evidence=source.result.evidence,
    )


def save_rabi(
    source: Completed[LenRabiCfg, RabiAcquisition],
    destination: Path,
    *,
    snapshot: RunSnapshot,
) -> None:
    """Write native length-Rabi data to an exact, not-yet-existing path.

    source supplies Engine's complete cfg/result pair; snapshot is historical
    entry state. Preserve acquisition evidence. Parent must exist; schema,
    existing-destination and I/O errors propagate, without retry or overwrite.
    """
    record: RunRecord[LenRabiCfg, LenRabiResult] = RunRecord(source.cfg, source.result)
    save_native(
        record,
        destination,
        spec=LenRabiExp().persistence_spec(),
        snapshot=snapshot,
        evidence=source.result.evidence,
    )


def save_readout(
    source: Completed[FreqGainCfg, ReadoutAcquisition],
    destination: Path,
    *,
    snapshot: RunSnapshot,
) -> None:
    """Write native frequency/gain SNR data to an exact new path.

    source supplies Engine's complete cfg/result pair; snapshot is historical
    entry state. Preserve native axes and acquisition evidence. Parent must
    exist; schema, existing-destination and I/O errors propagate.
    """
    record: RunRecord[FreqGainCfg, FreqGainResult] = RunRecord(
        source.cfg, source.result
    )
    save_native(
        record,
        destination,
        spec=FreqGainExp().persistence_spec(),
        snapshot=snapshot,
        evidence=source.result.evidence,
    )


def save_t1(
    source: Completed[T1Cfg, T1Acquisition],
    destination: Path,
    *,
    snapshot: RunSnapshot,
) -> None:
    """Write native T1 data to the workflow's exact, not-yet-existing path.

    source supplies the complete cfg/result pair; snapshot is historical entry
    state. Keep microsecond-to-disk mapping and acquisition evidence from their
    owners. Parent must exist; schema, collision and I/O errors propagate.
    """
    record: RunRecord[T1Cfg, T1Result] = RunRecord(source.cfg, source.result)
    save_native(
        record,
        destination,
        spec=T1Exp().persistence_spec(),
        snapshot=snapshot,
        evidence=source.result.evidence,
    )


def save_t2echo(
    source: Completed[T2EchoCfg, T2EchoAcquisition],
    destination: Path,
    *,
    snapshot: RunSnapshot,
) -> None:
    """Write zero-detune Echo to the workflow's exact, not-yet-existing native path.

    source owns cfg/result and fixed acquisition/device evidence. snapshot is
    historical entry/point/roles. Preserve native time-to-disk mapping and
    persistence declaration. Parent must exist; schema/collision/I/O errors
    propagate without recapture, layout selection or retry.
    """
    record: RunRecord[T2EchoCfg, T2EchoResult] = RunRecord(source.cfg, source.result)
    save_native(
        record,
        destination,
        spec=T2EchoExp().persistence_spec(),
        snapshot=snapshot,
        evidence=source.result.evidence,
    )

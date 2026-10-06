"""Adapter-owned acquisition identity and software-only environment evidence."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import socket
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Literal

from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.context import RunContext
from zcu_tools.resources.entry import new_run_id as generate_run_id


@dataclass(frozen=True)
class AcquisitionEvidence:
    """Facts retained with one adapter's successfully published acquisition.

    run_id is the UTC timestamp plus six hexadecimal random characters from
    new_run_id. started_at and finished_at are timezone-aware UTC boundaries
    around core acquisition and its error check, excluding presentation,
    analysis and saving. completion is complete for normal return without stop,
    or stopped for normal return with the existing StopSignal set.
    provenance contains software/host/cached-config facts captured before the
    acquisition; its version mapping is copied on construction.
    """

    run_id: str
    started_at: datetime
    finished_at: datetime
    completion: Literal["complete", "stopped"]
    provenance: SoftwareProvenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provenance",
            replace(
                self.provenance,
                software_versions=dict(self.provenance.software_versions),
            ),
        )


def new_run_id(started_at: datetime) -> str:
    """Return YYYYMMDDTHHMMSSZ-xxxxxx with a random six-hex-character suffix.

    started_at must be timezone-aware with zero UTC offset. Raise ValueError
    for naive or non-UTC datetimes; no clock or hardware is read.
    """
    if started_at.tzinfo is None or started_at.utcoffset() != timedelta(0):
        raise ValueError("started_at must be a timezone-aware UTC datetime")
    return generate_run_id(at=started_at)


def capture_software_provenance(
    *,
    context: RunContext | None = None,
    source_directory: Path | None = None,
) -> SoftwareProvenance:
    """Capture software, Git, host and optional cached SoC configuration facts.

    Read Python and installed zcu-tools/NumPy versions, optional QICK version,
    hostname, and the enclosing source checkout's commit/dirty state.
    source_directory defaults to this module's directory and must be a directory.
    Without an enclosing .git marker, Git fields are None and Git is not run.
    Only a genuinely absent QICK distribution produces qick_version=None.

    context may be absent or have an opaque soccfg without dump_cfg, producing
    soc_fingerprint=None. Otherwise hash the UTF-8 dump_cfg() JSON string with
    SHA256. Never call soc or devices or request live instrument information.
    Return a detached version mapping. Invalid inputs and failures of available
    version/Git/host/config capture propagate, rather than becoming None.
    Read-only Git commands disable optional index locks and have a ten-second
    timeout; timeout errors propagate.
    """
    directory = Path(__file__).parent if source_directory is None else source_directory
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    versions = {
        "python": platform.python_version(),
        "zcu-tools": metadata.version("zcu-tools"),
        "numpy": metadata.version("numpy"),
    }
    try:
        qick_version = metadata.version("qick")
    except metadata.PackageNotFoundError:
        qick_version = None
    commit, dirty = _git_evidence(directory)
    return SoftwareProvenance(
        software_versions=versions,
        git_commit=commit,
        git_dirty=dirty,
        qick_version=qick_version,
        soc_fingerprint=_soc_fingerprint(context),
        hostname=socket.gethostname(),
    )


def _git_evidence(directory: Path) -> tuple[str | None, bool | None]:
    """Read enclosing checkout commit/dirty facts; non-Git installs have neither."""
    for root in (directory, *directory.parents):
        if (root / ".git").exists():
            command = ["git", "--no-optional-locks", "-C", str(root)]
            commit = subprocess.run(
                [*command, "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            status = subprocess.run(
                [*command, "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            return commit, bool(status)
    return None, None


def _soc_fingerprint(context: RunContext | None) -> str | None:
    """Hash only cached dump_cfg JSON; an opaque context has no fingerprint."""
    if context is None:
        return None
    try:
        dump_cfg = context.soccfg.dump_cfg
    except AttributeError:
        # A declared descriptor failing is not an absent dynamic capability.
        if inspect.getattr_static(context.soccfg, "dump_cfg", None) is not None:
            raise
        return None
    serialized = dump_cfg()
    if not isinstance(serialized, str):
        raise TypeError("soccfg.dump_cfg() must return a JSON string")
    json.loads(serialized)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

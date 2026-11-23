"""Starting CZ asset fingerprints, independent of shared waveform validation."""

from hashlib import file_digest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import TypeAdapter
from zcu_tools.program.v2.modules import ArbWaveformCfg, FlatTopWaveformCfg

from .._drive import QubitDriveCfg

_FINGERPRINTS = TypeAdapter(dict[str, str])


def start_asset_paths(drive: QubitDriveCfg) -> tuple[str, ...]:
    """Return distinct file paths referenced by the two starting CZ pulses."""
    paths: dict[str, None] = {}
    for pulse in (drive.qubit1, drive.qubit2):
        waveform = pulse.waveform
        if isinstance(waveform, FlatTopWaveformCfg):
            waveform = waveform.raise_waveform
        if isinstance(waveform, ArbWaveformCfg):
            paths[str(Path(waveform.path).absolute())] = None
    return tuple(paths)


def _sha256(path: str) -> str:
    with Path(path).open("rb") as asset:
        return file_digest(asset, "sha256").hexdigest()


def capture_start_assets(drive: QubitDriveCfg) -> NDArray[np.uint8] | None:
    """Record SHA-256 of each original file; no snapshot for built-in pulses.

    Native grouped data accepts numeric arrays. Store the optional manifest as
    UTF-8 bytes so its diagnostic paths also survive an unavailable typed cfg.
    """
    paths = start_asset_paths(drive)
    if not paths:
        return None
    fingerprints = {path: _sha256(path) for path in paths}
    return np.frombuffer(_FINGERPRINTS.dump_json(fingerprints), dtype=np.uint8)


def start_asset_issues(
    snapshot: NDArray[np.uint8] | None, *, paths: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    """Diagnose saved assets, or require a matching fingerprint for given paths."""
    saved = (
        _FINGERPRINTS.validate_json(snapshot.tobytes()) if snapshot is not None else {}
    )
    issues = []
    for path in saved if paths is None else paths:
        expected = saved.get(path)
        if expected is None:
            reason = "no saved fingerprint"
        else:
            try:
                actual = _sha256(path)
            except FileNotFoundError:
                reason = "file missing"
            except OSError as error:
                reason = f"cannot read file ({error.strerror})"
            else:
                if actual == expected:
                    continue
                reason = "file content changed since Run (SHA-256 mismatch)"
        issues.append(f"Starting CZ asset {path}: {reason}")
    return tuple(issues)

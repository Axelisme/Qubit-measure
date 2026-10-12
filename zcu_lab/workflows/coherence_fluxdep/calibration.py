"""coherence_fluxdep calibration helpers for lab workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CalibrationFailed:
    """Rejected calibration stage and human-readable reason; no partial values.

    stage is frequency, rabi or readout. reason describes the Schedule failure
    or acceptance rejection. Exceptions are not converted into this value.
    """

    stage: Literal["frequency", "rabi", "readout"]
    reason: str


@dataclass(frozen=True)
class CalibrationFiles:
    """Native filenames returned only after a complete accepted calibration.

    frequency names completed spectroscopy attempts in widening order, relative
    to this invocation's root. rabi names its completed Length-Rabi acquisition.
    readout names the completed landscape, or None when cadence skips that stage.
    These references do not guarantee later file existence or completeness.
    """

    frequency: tuple[str, ...]
    rabi: str
    readout: str | None

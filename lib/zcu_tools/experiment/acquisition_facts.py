"""Generic observations from program acquisition and data-stop decisions."""

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field


@dataclass(frozen=True)
class StopDecision:
    """A data condition result and its optional finite scalar observation.

    met requests local stop when True. value has caller-defined units and meaning.
    None means no early stop, an unavailable value, or a nonfinite observation.
    The runtime retains value only when this decision ends acquisition early.
    bool(decision) returns met, so conditions remain usable in ordinary branches.
    """

    met: bool
    value: float | None = None

    def __bool__(self) -> bool:
        """Return whether the condition requests local stop."""
        return self.met


@dataclass(frozen=True)
class AcquisitionFacts:
    """One program acquisition, including all its producer retries.

    requested_rounds is the positive configured round count. completed_rounds
    counts completed round hooks in the last attempted producer call, not the
    sum of discarded retry data. attempts counts producer calls, including
    failures; zero means cancellation prevented a call. early_stopped means a
    data condition ended the final attempt before requested_rounds. stop_value
    is its finite scalar with caller-defined meaning and units. None means no
    early stop, an unavailable value, or a nonfinite observation.
    retry_reasons lists actual retries in order: "shot_overrun" for transport
    excess shots, otherwise the producer exception's class name. Empty means
    no retry; the final failure is not a retry reason.
    simulated_physical_s is simulated shot time in seconds for all rounds whose
    execute returned, including discarded retries, independent of pacing scale.
    None means a non-simulation acquisition. It excludes flux ramp/settle time
    and compile/upload overhead.
    """

    requested_rounds: int
    completed_rounds: int
    attempts: int
    early_stopped: bool
    stop_value: float | None = None
    retry_reasons: tuple[str, ...] = ()
    simulated_physical_s: Annotated[
        float | None, Field(exclude_if=lambda value: value is None)
    ] = None

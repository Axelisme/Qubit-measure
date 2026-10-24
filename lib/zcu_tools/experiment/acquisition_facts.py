"""Generic observations from program acquisition and data-stop decisions."""

from dataclasses import dataclass


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
    """

    requested_rounds: int
    completed_rounds: int
    attempts: int
    early_stopped: bool
    stop_value: float | None = None

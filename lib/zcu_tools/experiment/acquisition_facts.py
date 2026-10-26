"""Data-driven stop decisions supplied by experiment authors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StopDecision:
    """A data condition result and its optional finite scalar observation.

    met requests local stop when True. value has caller-defined units and meaning.
    None means no early stop, an unavailable value, or a nonfinite observation.
    The builder logs value only when this decision ends acquisition early.
    bool(decision) returns met, so conditions remain usable in ordinary branches.
    """

    met: bool
    value: float | None = None

    def __bool__(self) -> bool:
        """Return whether the condition requests local stop."""
        return self.met

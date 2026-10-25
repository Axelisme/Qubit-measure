"""Author-owned selector inputs and scalar report mechanics for mux recipes."""

from pydantic import Field, model_validator
from zcu_tools.gui.form import FormModel


class ToneRangeInputs(FormModel):
    """One tone's finite start/stop in experiment units, without a point count.

    Frequency uses MHz, gain in a.u. within [-1, 1] and length us. Supply all tones
    in original list order. Only start and stop are accepted; n is shared.
    """

    start: float = Field(description="First value in the experiment's units.")
    stop: float = Field(description="Last value in the experiment's units.")

    @model_validator(mode="before")
    @classmethod
    def reject_per_tone_count(cls, value: object) -> object:
        """Explain the range-only contract when a legacy tone count is supplied."""
        if isinstance(value, dict) and "n" in value:
            raise ValueError(
                "Tone sweep fields are start, stop only; use top-level n for all tones"
            )
        return value


class MuxToneReport(FormModel):
    """Identify one native analysis outcome without changing its row order.

    index is the original complete-list position. name comes from that same
    analysis capture, or is null when names are unavailable. failure preserves
    the fitter's reason; failed rows have no estimated scalar values.
    """

    index: int = Field(description="Original complete-list tone position.")
    name: str | None = Field(description="Captured qubit name, or null without names.")
    failure: str | None = Field(
        default=None, description="Native fit failure reason, if any."
    )

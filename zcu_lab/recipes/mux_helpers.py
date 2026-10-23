"""Author-owned selector inputs and scalar report mechanics for mux recipes."""

from pydantic import Field, model_validator
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import FieldEdit, RecipeTab


class ToneRangeInputs(FormModel):
    """One tone's finite start/stop in experiment units, without a point count.

    Frequency uses MHz, gain normalized units and length us. Supply all tones
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


def mux_sweep_edits(
    tab: RecipeTab, sweeps: list[ToneRangeInputs] | None, n: int | None
) -> list[FieldEdit]:
    """Build range and shared-count edits for a captured full tone list.

    Null retains the corresponding Form inputs. sweeps must cover every tone
    in original order, including nonparticipants; a length mismatch raises
    ValueError before edits are installed. n is the validated shared count.
    Each supplied range replaces the complete nullable sweep, including None
    on a nonparticipant. This reads the Form without changing it. The caller
    applies these edits before overrides in the same validated batch.
    """
    edits: list[FieldEdit] = []
    if sweeps is not None:
        tones = tab.read(("tones",)).value
        if not isinstance(tones, list):
            raise ValueError("tones must be an available full tone list")
        if len(sweeps) != len(tones):
            raise ValueError(
                f"sweeps length={len(sweeps)}; expected {len(tones)} tones, each with start, stop; use top-level n"
            )
        for index, sweep in enumerate(sweeps):
            edits.append(FieldEdit(path=("tones", index, "sweep"), value=sweep))
    edits.append(FieldEdit(path=("expts",), value=n))
    return edits


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

"""Shared inputs, reports and picker updates for mux DC spectroscopy."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StrictInt, TypeAdapter
from typing_extensions import TypedDict
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeTab,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from .mux_helpers import ToneRangeInputs


class MuxFluxInputs(RecipeInputs):
    """Omitted Form inputs retain values; null clears them; overrides apply last."""

    participants: list[StrictInt] | None | Keep = Field(
        default=KEEP,
        description="Unique original tone indices in acquisition order; omission keeps the Form selection; null clears it.",
    )
    flux_unit: Literal["A", "V", "native"] | None | Keep = Field(
        default=KEEP,
        description="DC device unit label; omission keeps the Form unit; null clears it. No conversion.",
    )
    flux_n: Annotated[int, Field(strict=True, gt=0)] | None | Keep = Field(
        default=KEEP,
        description="Shared DC point count; omission keeps Expts; null clears it. Local lines advance in a zip, not a product.",
    )
    flux_half: dict[str, float] | None | Keep = Field(
        default=KEEP,
        description="Initial half-flux line per participant Q in native DC units; omission or null sends per-Q reset commands to use Run capture.",
    )
    flux_integer: dict[str, float] | None | Keep = Field(
        default=KEEP,
        description="Initial integer-flux line per participant Q in native DC units; omission or null sends per-Q reset commands to use Run capture. Owner field: flux_int.",
    )


class MuxOnetoneFluxInputs(MuxFluxInputs):
    """Common MHz frequency offsets for all original readout tones; n → df Expts."""

    df: ToneRangeInputs | None | Keep = Field(
        default=KEEP,
        description="Common start/stop offsets in MHz added to each base tone frequency; omission keeps the Form range; null clears its bounds.",
    )
    n: Annotated[int, Field(strict=True, gt=0)] | None | Keep = Field(
        default=KEEP,
        description="Shared frequency point count, n → frequency.df Expts; omission keeps the Form count; null clears it.",
    )


class MuxTwotoneFluxInputs(MuxFluxInputs):
    """Absolute MHz drive bounds for every original tone; n → frequency.expts."""

    sweeps: list[ToneRangeInputs] | None | Keep = Field(
        default=KEEP,
        description="Absolute MHz start/stop for every original tone, including nonparticipants; omission keeps ranges; null clears their bounds. No per-tone n.",
    )
    n: Annotated[int, Field(strict=True, gt=0)] | None | Keep = Field(
        default=KEEP,
        description="Shared hardware frequency count, n → frequency.expts; omission keeps Expts; null clears it.",
    )


class FluxToneIdentity(BaseModel):
    """Decode the fixed complete-list Q identity from a Form observation."""

    name: str = Field(description="Q name at this original full-list position.")


class MuxFluxAnalysisUpdate(TypedDict, closed=True):
    """Same-operation picker updates; null lines restore this Q's Run capture."""

    tone: int
    """Original index in the complete fixed tones list, not participant order."""
    flux_half: float | None
    """Initial half-flux line in native DC units; null uses Run capture."""
    flux_int: float | None
    """Initial integer-flux line in native DC units; null uses Run capture."""


class MuxFluxToneReport(RecipeReport):
    """Native ordered estimates/failures and the confirmed writeback outcome."""

    index: int = Field(description="Owner's original full-list tone index.")
    name: str = Field(description="Owner's captured Q name.")
    flux_half: float | None = Field(description="Half-flux value in native DC units.")
    flux_integer: float | None = Field(
        description="Integer-flux value in native DC units."
    )
    flux_period: float | None = Field(description="Period in native DC units.")


class MuxFluxReport(RecipeReport):
    """Native ordered estimates/failures and the confirmed writeback outcome."""

    tones: list[MuxFluxToneReport] = Field(
        description="Completed picker outcomes in participant order, retaining original indices."
    )


def mux_flux_analysis_updates(
    tab: RecipeTab, inputs: MuxFluxInputs
) -> list[MuxFluxAnalysisUpdate]:
    """Read final validated selection and build same-operation updates in Q order.

    Original indices come from the complete fixed tones list. Unknown picker
    keys raise needs_parameters before acquisition. Both line keys are always
    sent: null restores Run-capture policy rather than retaining the previous
    participant's explicit line. These per-Q reset commands are required even
    when the caller omits a map; they are not omitted caller field edits.
    This never changes Run cfg or calibration.
    """
    tones = TypeAdapter(list[FluxToneIdentity]).validate_python(
        tab.read(("tones",)).value
    )
    participants = TypeAdapter(list[int]).validate_python(
        tab.read(("participants",)).value
    )
    names = [tones[index].name for index in participants]
    flux_half = None if isinstance(inputs.flux_half, Keep) else inputs.flux_half
    flux_integer = (
        None if isinstance(inputs.flux_integer, Keep) else inputs.flux_integer
    )
    for field, values in (
        ("flux_half", flux_half),
        ("flux_integer", flux_integer),
    ):
        unknown = set(values or ()) - set(names)
        if unknown:
            raise RecipeNeedsParameters(
                (
                    MissingParameter(
                        field,
                        f"Picker keys must name selected participants {names!r}; unknown keys: {sorted(unknown)!r}",
                    ),
                )
            )
    return [
        {
            "tone": index,
            "flux_half": (flux_half or {}).get(name),
            "flux_int": (flux_integer or {}).get(name),
        }
        for index, name in zip(participants, names, strict=True)
    ]

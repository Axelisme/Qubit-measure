"""Shared inputs and flat-cfg edits for the four mux DC spectroscopy recipes."""

from typing import Literal

from pydantic import BaseModel, Field, StrictInt, TypeAdapter
from typing_extensions import TypedDict
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
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
    """Experiment inputs; null retains captured Form values, overrides apply last."""

    participants: list[StrictInt] | None = Field(
        None,
        description="Unique original tone indices in acquisition order; null keeps the captured Form selection.",
    )
    flux_unit: Literal["A", "V", "native"] | None = Field(
        None,
        description="DC device unit label; null keeps the Form unit. No conversion.",
    )
    flux_n: int | None = Field(
        None,
        strict=True,
        gt=0,
        description="Shared DC point count; null keeps Expts. Local lines advance in a zip, not a product.",
    )
    flux_half: dict[str, float] | None = Field(
        None,
        description="Optional initial half-flux line per participant Q, in native DC units; absent Q uses Run capture.",
    )
    flux_integer: dict[str, float] | None = Field(
        None,
        description="Optional initial integer-flux line per participant Q, in native DC units; absent Q uses Run capture. Sent to owner as flux_int.",
    )


class MuxOnetoneFluxInputs(MuxFluxInputs):
    """Common MHz frequency offsets for all original readout tones; n → df Expts."""

    df: ToneRangeInputs | None = Field(
        None,
        description="Common start/stop offsets in MHz added to each base tone frequency; null keeps the Form range.",
    )
    n: int | None = Field(
        None,
        strict=True,
        gt=0,
        description="Shared frequency point count, n → frequency.df Expts; null keeps the Form count.",
    )


class MuxTwotoneFluxInputs(MuxFluxInputs):
    """Absolute MHz drive bounds for every original tone; n → frequency.expts."""

    sweeps: list[ToneRangeInputs] | None = Field(
        None,
        description="Absolute MHz start/stop for every original tone, including nonparticipants; null keeps ranges. No per-tone n.",
    )
    n: int | None = Field(
        None,
        strict=True,
        gt=0,
        description="Shared hardware frequency count, n → frequency.expts; null keeps Expts.",
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


def flux_range_edits(
    tab: RecipeTab, sweeps: list[ToneRangeInputs] | None, path: tuple[str, ...]
) -> list[FieldEdit]:
    """Build full-list start/stop edits at path, rejecting partial range lists.

    path addresses tones or frequency.tones in the flat cfg. Null keeps all
    ranges. A supplied list must cover the complete original list, including
    nonparticipants. Length mismatch raises needs_parameters before Run.
    """
    if sweeps is None:
        return []
    tones = TypeAdapter(list[FluxToneIdentity]).validate_python(tab.read(path).value)
    if len(sweeps) != len(tones):
        raise RecipeNeedsParameters(
            (
                MissingParameter(
                    ".".join(path),
                    f"Provide {len(tones)} start/stop ranges in original tone order, including nonparticipants; received {len(sweeps)}.",
                ),
            )
        )
    return [
        FieldEdit(path=(*path, index, "sweep", endpoint), value=value)
        for index, sweep in enumerate(sweeps)
        for endpoint, value in (("start", sweep.start), ("stop", sweep.stop))
    ]


def onetone_flux_edits(inputs: MuxOnetoneFluxInputs) -> list[FieldEdit]:
    """Map common MHz offsets and n to frequency.df in the flat Form cfg."""
    edits = [FieldEdit(path=("frequency", "df", "n"), value=inputs.n)]
    if inputs.df is not None:
        edits.extend(
            (
                FieldEdit(path=("frequency", "df", "start"), value=inputs.df.start),
                FieldEdit(path=("frequency", "df", "stop"), value=inputs.df.stop),
            )
        )
    return edits


def global_flux_edits(flux: ToneRangeInputs | None, n: int | None) -> list[FieldEdit]:
    """Map native global DC endpoints and shared flux_n to flux Expts, no conversion."""
    edits = [FieldEdit(path=("flux", "n"), value=n)]
    if flux is not None:
        edits.extend(
            (
                FieldEdit(path=("flux", "start"), value=flux.start),
                FieldEdit(path=("flux", "stop"), value=flux.stop),
            )
        )
    return edits


def mux_flux_analysis_updates(
    tab: RecipeTab, inputs: MuxFluxInputs
) -> list[MuxFluxAnalysisUpdate]:
    """Read final validated selection and build same-operation updates in Q order.

    Original indices come from the complete fixed tones list. Unknown picker
    keys raise needs_parameters before acquisition. Both line keys are always
    sent: null restores Run-capture policy rather than retaining the previous
    participant's explicit line. This never changes Run cfg or calibration.
    """
    tones = TypeAdapter(list[FluxToneIdentity]).validate_python(
        tab.read(("tones",)).value
    )
    participants = TypeAdapter(list[int]).validate_python(
        tab.read(("participants",)).value
    )
    names = [tones[index].name for index in participants]
    for field, values in (
        ("flux_half", inputs.flux_half),
        ("flux_integer", inputs.flux_integer),
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
            "flux_half": (inputs.flux_half or {}).get(name),
            "flux_int": (inputs.flux_integer or {}).get(name),
        }
        for index, name in zip(participants, names, strict=True)
    ]

"""Shared inputs and flat-cfg edits for the four mux DC spectroscopy recipes."""

from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import TypedDict
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import FieldEdit, RecipeTab
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from .mux_helpers import MuxInline, MuxLibrary, ToneRangeInputs, mux_overrides


class MuxFluxInputs(FormModel):
    """Select captured sources and optional native-unit picker lines by Q name.

    Null fields retain Form inputs. Picker maps may omit any participant; those
    lines use the Run capture. Keys must name final selected participants.
    Overrides apply last and use original full-list indices, never row order.
    """

    target: str | None = Field(
        None, min_length=1, description="Mux Reset alias; null uses global aliases."
    )
    reuse_tab_id: str | None = Field(
        None, min_length=1, description="Idle matching tab to Reset; null opens a tab."
    )
    participants: list[str] | None = Field(
        None,
        description="Unique Q names in acquisition order; null keeps the Form selection.",
    )
    library: list[MuxLibrary] | None = Field(
        None, description="Ordered captured module selections; null retains sources."
    )
    inline: list[MuxInline] | None = Field(
        None,
        description="Ordered captured-module copies, ancestors first; null retains sources.",
    )
    overrides: list[FieldEdit] | None = Field(
        None,
        description="Ordered scalar edits applied last at full original tone positions.",
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


class MuxFluxPickSummary(BaseModel):
    """Owner's accepted native DC lines and original Q identity, without defaults."""

    tone: int = Field(description="Original acquired tone index, not participant row.")
    qubit: str = Field(description="Q name from this Run's full cfg at tone.")
    flx_half: float | None = Field(
        description="Picked half-flux native DC value, or null if unavailable."
    )
    flx_int: float | None = Field(
        description="Picked integer-flux native DC value, or null if unavailable."
    )
    flx_period: float | None = Field(
        description="Native DC flux period, or null if unavailable."
    )


class MuxFluxToneReport(FormModel):
    """One participant's selected lines and confirmed writeback outcome."""

    index: int = Field(description="Owner's original full-list tone index.")
    name: str = Field(description="Owner's captured Q name.")
    flux_half: float | None = Field(description="Half-flux value in native DC units.")
    flux_integer: float | None = Field(
        description="Integer-flux value in native DC units."
    )
    flux_period: float | None = Field(description="Period in native DC units.")
    writeback_applied: bool = Field(
        description="True only if the owner receipt confirms selected items."
    )


class MuxFluxReport(FormModel):
    """Ordered completed participants; Save output statuses are separate in runs."""

    tones: list[MuxFluxToneReport] = Field(
        description="Completed picker outcomes in participant order, retaining original indices."
    )
    analysis_failure: str | None = Field(
        default=None,
        description="Owner failure reason; null when all participant analyses completed.",
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


def mux_flux_final_edits(inputs: MuxFluxInputs) -> list[FieldEdit]:
    """Append participant/unit retention edits, then caller's ordered overrides."""
    return [
        FieldEdit(
            path=("participants",),
            value=None
            if inputs.participants is None
            else [name for name in inputs.participants],
        ),
        FieldEdit(path=("flux_unit",), value=inputs.flux_unit),
        *mux_overrides(inputs.overrides),
    ]


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
    participants = TypeAdapter(list[str]).validate_python(
        tab.read(("participants",)).value
    )
    names = [tone.name for tone in tones]
    for field, values in (
        ("flux_half", inputs.flux_half),
        ("flux_integer", inputs.flux_integer),
    ):
        unknown = set(values or ()) - set(participants)
        if unknown:
            raise RecipeNeedsParameters(
                (
                    MissingParameter(
                        field,
                        f"Picker keys must name selected participants {participants!r}; unknown keys: {sorted(unknown)!r}",
                    ),
                )
            )
    return [
        {
            "tone": names.index(name),
            "flux_half": (inputs.flux_half or {}).get(name),
            "flux_int": (inputs.flux_integer or {}).get(name),
        }
        for name in participants
    ]

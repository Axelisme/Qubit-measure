"""Author-owned selector inputs and scalar report mechanics for mux recipes."""

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import Expr, FieldEdit, FormPath


class MuxLibrary(FormModel):
    """Select a captured module key at an original-position Form path.

    path addresses the complete tone list, never participant row order. A null
    key keeps the GUI selection. The GUI checks selector and source validity.
    """

    path: FormPath = Field(
        description="Selector path; integers are full tone positions."
    )
    key: str | None = Field(description="Module key; null keeps the GUI value.")


class MuxInline(FormModel):
    """Copy a selector's captured module for local edits without refreshing it.

    path uses original full-tone positions. List referenced ancestors before
    their children; missing or invalid selections fail at the GUI owner.
    """

    path: FormPath = Field(description="Selector path to copy; ancestors come first.")


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


def mux_overrides(edits: list[FieldEdit] | None) -> list[FieldEdit]:
    """Copy ordered scalar edits, labeling Expr values as derived.

    Null or an empty list supplies no overrides. Paths and values stay authored;
    null values keep GUI inputs. Literal values retain the baseline explicit
    provenance. This function does not apply edits or validate calibrations.
    """
    return [
        FieldEdit(
            path=edit.path,
            value=edit.value,
            source="derived" if isinstance(edit.value, Expr) else "explicit",
        )
        for edit in edits or ()
    ]


class MuxSummaryTone(BaseModel):
    """Owner summary row identity, independent of participant row order.

    position is the original complete-list index. name is the captured name,
    or null when the analysis source has no names.
    """

    position: int
    name: str | None


class MuxSummaryFailure(MuxSummaryTone):
    """Owner fit failure at a captured tone; failure is its unchanged reason."""

    failure: str

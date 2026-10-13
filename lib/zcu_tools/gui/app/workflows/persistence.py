"""App-owned workflow input memento; not a native run or journal snapshot."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from zcu_tools.gui.form import FormPath


class WorkflowInputSnapshot(BaseModel):
    """Versioned parsed input, retaining expressions without evaluating them.

    version identifies this app memento format and must be 2. plan and knobs
    contain detached FormInput JSON trees, including missing/null and expression
    tags. Empty mappings mean fresh-default overlays. No run or hardware state
    is saved. Unknown top-level fields and malformed trees are rejected.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    version: Literal[2] = 2
    plan: dict[str, JsonValue] = Field(default_factory=dict)
    knobs: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowInputRestoreReport:
    """Paths discarded during restoration by the current forms and capture.

    plan_dropped and knobs_dropped are model-field tuple paths in the respective
    form: removed or type-incompatible fields, and restored leaves whose captured
    source check failed (for example a module ref without a captured library).
    An empty tuple means every supplied field was compatible.
    """

    plan_dropped: tuple[FormPath, ...]
    knobs_dropped: tuple[FormPath, ...]


class WorkflowInputCodec:
    """Validate the app memento at the shared caretaker's JSON boundary."""

    def default(self) -> WorkflowInputSnapshot:
        """Return empty overlays so the Window supplies current form defaults."""
        return WorkflowInputSnapshot()

    def decode(self, raw: object) -> WorkflowInputSnapshot:
        """Decode parsed JSON; unknown versions/fields or invalid shapes raise."""
        return WorkflowInputSnapshot.model_validate(raw)

    def encode(self, state: WorkflowInputSnapshot) -> object:
        """Return a detached JSON snapshot, preserving expression envelopes."""
        return state.model_dump(mode="json")

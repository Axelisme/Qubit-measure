"""App-owned workflow input memento; not a native run or journal snapshot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormPath
from zcu_tools.gui.session.persistence import SnapshotIdentityError


class WorkflowInputSnapshot(BaseModel):
    """Versioned parsed input, retaining expressions without evaluating them.

    version identifies this app memento format and must be 3. plan and knobs
    contain complete versioned FormInput envelopes, including independent
    selections, raw branches, missing/null and expressions. None means fresh
    defaults only when no saved draft exists. No run or hardware state is saved.
    Unknown top-level fields and old versions are rejected, never migrated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    version: Literal[3] = 3
    plan: dict[str, JsonValue] | None = None
    knobs: dict[str, JsonValue] | None = None

    @classmethod
    def capture(
        cls, plan: FormInput[FormModel], knobs: FormInput[FormModel]
    ) -> WorkflowInputSnapshot:
        """Capture both opaque FormInput envelopes without evaluating them."""
        adapter = TypeAdapter(dict[str, JsonValue])
        return cls(
            plan=adapter.validate_json(plan.to_json()),
            knobs=adapter.validate_json(knobs.to_json()),
        )

    def inputs(
        self, plan: type[FormModel], knobs: type[FormModel], env: FormEnv
    ) -> tuple[FormInput[FormModel], FormInput[FormModel]]:
        """Prepare both trusted drafts before either host state is changed.

        Saved envelopes restore whole input, never defaults overlays. Annotation
        identity mismatch raises SnapshotIdentityError before host mutation. Missing sources and current option
        membership are ordinary result errors, not restoration failures. env is
        used only for fresh defaults when an envelope is absent.
        """
        return (
            plan.default_input(env)
            if self.plan is None
            else _restore_envelope(plan, self.plan),
            knobs.default_input(env)
            if self.knobs is None
            else _restore_envelope(knobs, self.knobs),
        )


def _restore_envelope(
    model: type[FormModel], envelope: dict[str, JsonValue]
) -> FormInput[FormModel]:
    try:
        return FormInput.from_json(model, json.dumps(envelope))
    except ValueError as exc:
        raise SnapshotIdentityError(str(exc)) from exc


@dataclass(frozen=True)
class WorkflowInputRestoreReport:
    """Successful restoration report retained for the host message seam.

    plan_dropped and knobs_dropped are empty: whole-draft restoration never
    discards paths. Identity failures raise before either form is installed.
    """

    plan_dropped: tuple[FormPath, ...]
    knobs_dropped: tuple[FormPath, ...]


class WorkflowInputCodec:
    """Validate the app memento at the shared caretaker's JSON boundary."""

    def default(self) -> WorkflowInputSnapshot:
        """Use current defaults when no persisted draft is available."""
        return WorkflowInputSnapshot()

    def decode(self, raw: object) -> WorkflowInputSnapshot:
        """Decode parsed JSON; unknown versions/fields or invalid shapes raise."""
        snapshot = WorkflowInputSnapshot.model_validate(raw)
        if snapshot.plan is None or snapshot.knobs is None:
            raise ValueError("Saved workflow input requires both FormInput envelopes")
        return snapshot

    def encode(self, state: WorkflowInputSnapshot) -> object:
        """Return a detached JSON snapshot, preserving expression envelopes."""
        return state.model_dump(mode="json")

"""App-owned plan/knobs memento, each using the shared FormInput JSON codec."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormPath


class WorkflowInputSnapshot(BaseModel):
    """Two independent parsed drafts; no run, hardware or declaration metadata.

    plan and knobs contain FormInput {data,selections} JSON objects. None means
    no saved draft, so the host uses fresh defaults for that form. Raw invalid
    input, missing/null, expressions and explicit selections remain authored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    plan: dict[str, JsonValue] | None = None
    knobs: dict[str, JsonValue] | None = None

    @classmethod
    def capture(
        cls, plan: FormInput[FormModel], knobs: FormInput[FormModel]
    ) -> WorkflowInputSnapshot:
        """Capture both shared FormInput JSON objects without evaluating them."""
        adapter = TypeAdapter(dict[str, JsonValue])
        return cls(
            plan=adapter.validate_json(plan.to_json()),
            knobs=adapter.validate_json(knobs.to_json()),
        )

    def inputs(
        self, plan: type[FormModel], knobs: type[FormModel], env: FormEnv
    ) -> tuple[FormInput[FormModel], FormInput[FormModel]]:
        """Decode both authored drafts before host mutation; absent drafts use defaults.

        Invalid JSON wire shapes raise ValueError. The host applies the common
        recursive restore_input mechanism to both results. Source/provider
        failures from fresh defaults propagate unchanged.
        """
        return (
            plan.default_input(env)
            if self.plan is None
            else FormInput.from_json(plan, json.dumps(self.plan)),
            knobs.default_input(env)
            if self.knobs is None
            else FormInput.from_json(knobs, json.dumps(self.knobs)),
        )


@dataclass(frozen=True)
class WorkflowInputRestoreReport:
    """Discarded raw/override paths from the same recursive restore on both forms.

    plan_dropped and knobs_dropped identify removed fields, incompatible shapes
    or selections the current annotation no longer accepts. Hosts display them.
    """

    plan_dropped: tuple[FormPath, ...]
    knobs_dropped: tuple[FormPath, ...]


class WorkflowInputCodec:
    """Validate the app's two FormInput objects at the caretaker JSON boundary."""

    def default(self) -> WorkflowInputSnapshot:
        """Use current defaults when no persisted draft is available."""
        return WorkflowInputSnapshot()

    def decode(self, raw: object) -> WorkflowInputSnapshot:
        """Decode only current FormInput wire shapes; format errors raise ValueError."""
        snapshot = WorkflowInputSnapshot.model_validate(raw)
        for draft in (snapshot.plan, snapshot.knobs):
            if draft is not None:
                FormInput.from_json(FormModel, json.dumps(draft))
        return snapshot

    def encode(self, state: WorkflowInputSnapshot) -> object:
        """Return detached plan/knobs JSON, preserving authored input and overrides."""
        return state.model_dump(mode="json")

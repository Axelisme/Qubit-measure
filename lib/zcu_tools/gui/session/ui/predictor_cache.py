"""Versioned, dialog-owned simulator cache; never a component document."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from zcu_tools.simulate.fluxonium.predict import FluxoniumParams


class PredictorSnapshot(BaseModel):
    """User trial in GHz/native units, including unknown inputs.

    version identifies the cache schema, model identifies the physics family,
    and params is a detached trial. Invalid physics is allowed until computation.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    model: Literal["fluxonium"] = "fluxonium"
    params: FluxoniumParams = Field(default_factory=FluxoniumParams)


class PredictorSnapshotCodec:
    """Validate JSON cache snapshots; unsupported versions raise ValueError.

    SingleFileCaretaker owns I/O and reports decode failures while restoring
    an empty trial. This codec never reads or writes component storage.
    """

    def default(self) -> PredictorSnapshot:
        """Return a fresh trial with all inputs unknown."""
        return PredictorSnapshot()

    def decode(self, raw: object) -> PredictorSnapshot:
        """Validate one snapshot; malformed data raises pydantic ValidationError."""
        return PredictorSnapshot.model_validate(raw)

    def encode(self, state: PredictorSnapshot) -> object:
        """Return the versioned JSON representation, retaining None fields."""
        return state.model_dump(mode="json")

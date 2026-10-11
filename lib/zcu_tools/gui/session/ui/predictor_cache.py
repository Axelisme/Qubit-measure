"""Versioned, dialog-owned simulator cache; never a component document."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.parameters import ModelName, ModelParams, params_class


class PredictorSnapshot(BaseModel):
    """User trial in GHz/native units, including unknown inputs.

    version identifies the cache schema, model identifies the physics family,
    and params is a detached trial. Invalid physics is allowed until computation.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    model: ModelName = "fluxonium"
    params: ModelParams = Field(default_factory=FluxoniumParams)

    @model_validator(mode="before")
    @classmethod
    def _decode_params_for_model(cls, data: object) -> object:
        """Decode a raw params object with the type selected by ``model``."""
        if not isinstance(data, dict):
            return data
        model = data.get("model", "fluxonium")
        params = data.get("params")
        if isinstance(params, dict) and isinstance(model, str):
            data = {**data, "params": params_class(model).from_mapping(params)}
        return data

    @model_validator(mode="after")
    def _params_match_model(self) -> Self:
        if self.params.model != self.model:
            raise ValueError(f"params do not belong to model {self.model!r}")
        return self


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

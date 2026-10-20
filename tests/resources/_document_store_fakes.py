from pydantic import BaseModel, ConfigDict, Field


class SyntheticDocument(BaseModel):
    format: str
    format_version: str
    values: dict[str, float]


class KnownValues(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: float = Field(ge=0)
    right: float


class StrictDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: str
    format_version: str
    values: KnownValues

"""Raw GE shot count and workflow spacing policy tunables."""

from pydantic import BaseModel, ConfigDict, Field


class SingleshotTunables(BaseModel):
    """Choose flux spacing and raw shots per preparation after accepted Rabi.

    min_spacing is the absolute distance from the last acquired GE point in
    plan flux units; zero allows every
    eligible point. shots counts each probe-off/on cloud. Defaults are
    provisional simulation values, not calibrated hardware settings.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    min_spacing: float = Field(
        default=0.0, ge=0, description="Minimum GE spacing in plan flux units"
    )
    shots: int = Field(default=100000, ge=1, description="Shots per probe-off/on cloud")

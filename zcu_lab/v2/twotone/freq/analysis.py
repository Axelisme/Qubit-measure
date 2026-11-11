"""Analysis options for the two-tone frequency experiment."""

from typing import Annotated, Literal

from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.cfg_model import FieldMeta


@validated_dataclass(frozen=True)
class FreqAnalyzeOptions:
    """Select lor/sinc and a candidate near expected_center_mhz (None: strongest).

    min_prominence is relative to projected signal ptp; min_distance_mhz is the
    minimum candidate separation. min_snr gates prominence and height above
    the median in first-difference MAD noise units. min_width_samples gates
    observable half-prominence width in sample intervals; 0 disables this gate.
    Raise it only for fine scans. peaks reports height SNR, not prominence SNR.
    The fitted overlay is always drawn.
    """

    model_type: Annotated[Literal["lor", "sinc"], FieldMeta(label="Model type")] = "lor"
    expected_center_mhz: Annotated[
        float | None,
        FieldMeta(label="Expected center", unit="MHz"),
        Field(allow_inf_nan=False),
    ] = None
    min_prominence: Annotated[
        float, FieldMeta(label="Minimum prominence / signal range"), Field(ge=0, le=1)
    ] = 0.2
    min_snr: Annotated[
        float,
        FieldMeta(label="Minimum height above median / noise sigma"),
        Field(ge=0, allow_inf_nan=False),
    ] = 5.0
    min_width_samples: Annotated[
        float,
        FieldMeta(label="Minimum half-prominence width", unit="samples"),
        Field(ge=0, allow_inf_nan=False),
    ] = 0.0
    min_distance_mhz: Annotated[
        float,
        FieldMeta(label="Minimum peak separation", unit="MHz"),
        Field(ge=0, allow_inf_nan=False),
    ] = 0.0

"""Readout calibration data, independent of acquisition and file I/O."""

from math import isfinite
from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from zcu_tools.core.complex_codec import ComplexTag, decode_complex, encode_complex

from .base import ConfigBase
from .field_meta import FieldMeta


class CircleClassifier(ConfigBase):
    """Circle centers in normalized ADC units, without IQ-offset subtraction.

    mode is the fixed discriminator "circle". g_center/e_center are the complex
    ground/excited centers: raw integrated ADC values divided by integration
    length, with no IQ offset removed. Both components must be finite. Complex
    values serialize as {"__complex__": [real, imaginary]} in the same units.
    radius is finite and positive in those units; None means Primary centers
    exist but the Post radius calibration has not been supplied.
    confusion_matrix is an optional finite 3x3 matrix: rows are true (prepared)
    states, columns are measured states, both ordered g/e/other. None means no
    population correction is available. Values are stored without transposition,
    range checks or row normalization.
    """

    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    state_names: ClassVar[tuple[str, ...]] = ("g", "e", "other")
    """State names in classification_processor output-axis order and length."""

    mode: Literal["circle"] = "circle"
    g_center: complex
    e_center: complex
    radius: Annotated[float, Field(gt=0)] | None = None
    confusion_matrix: (
        Annotated[
            list[Annotated[list[float], Field(min_length=3, max_length=3)]],
            Field(min_length=3, max_length=3),
        ]
        | None
    ) = None

    @field_validator("g_center", "e_center", mode="before")
    @classmethod
    def _decode_center(cls, value: object) -> object:
        return decode_complex(value) if isinstance(value, dict) else value

    @field_validator("g_center", "e_center")
    @classmethod
    def _finite_center(cls, value: complex) -> complex:
        if not isfinite(value.real) or not isfinite(value.imag):
            raise ValueError(
                "Classifier centers must have finite real and imaginary parts"
            )
        return value

    @field_serializer("g_center", "e_center")
    def _encode_center(self, value: complex) -> ComplexTag:
        return encode_complex(value)


class LinearClassifier(ConfigBase):
    """Two-state threshold calibration in normalized, offset-free ADC units.

    mode is the fixed discriminator "linear". angle is finite radians, rotating
    normalized complex IQ by exp(1j*angle), in the GE analysis theta direction.
    threshold is the finite real-axis boundary after that rotation, in raw
    integrated ADC units divided by integration length, without subtracting IQ
    offset. confusion_matrix is an optional finite 2x2 matrix: rows are true
    (prepared) states and columns measured states, both ordered g/e. None means
    no population correction is available. Values are stored without
    transposition, range checks or row normalization. There is no other state.
    """

    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    state_names: ClassVar[tuple[str, ...]] = ("g", "e")
    """State names in classification_processor output-axis order and length."""

    mode: Literal["linear"] = "linear"
    angle: Annotated[float, FieldMeta(unit="rad")]
    threshold: float
    confusion_matrix: (
        Annotated[
            list[Annotated[list[float], Field(min_length=2, max_length=2)]],
            Field(min_length=2, max_length=2),
        ]
        | None
    ) = None


class Centroid3Classifier(ConfigBase):
    """Three-state nearest-center calibration in mean-sample ADC coordinates.

    mode is fixed to "centroid3". Required g_center/e_center/f_center are finite,
    pairwise distinct complex centers in raw integrated ADC / rounded integration
    sample count, without IQ-offset subtraction. Complex values serialize as
    {"__complex__": [real, imaginary]}. Required confusion_matrix is finite 3x3:
    prepared g/e/f rows, measured g/e/f columns. Values are stored unchanged,
    without transposition, normalization, range or row-sum restrictions.
    Classification chooses the Euclidean nearest center, ties in g/e/f order,
    with no rejection region. Acquire stores [g,e]; the third fraction is f.
    """

    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    state_names: ClassVar[tuple[str, ...]] = ("g", "e", "f")
    """State names in classification_processor output-axis order and length."""

    mode: Literal["centroid3"] = "centroid3"
    g_center: complex
    e_center: complex
    f_center: complex
    confusion_matrix: Annotated[
        list[Annotated[list[float], Field(min_length=3, max_length=3)]],
        Field(min_length=3, max_length=3),
    ]

    @field_validator("g_center", "e_center", "f_center", mode="before")
    @classmethod
    def _decode_center(cls, value: object) -> object:
        return decode_complex(value) if isinstance(value, dict) else value

    @field_validator("g_center", "e_center", "f_center")
    @classmethod
    def _finite_center(cls, value: complex) -> complex:
        if not isfinite(value.real) or not isfinite(value.imag):
            raise ValueError(
                "Classifier centers must have finite real and imaginary parts"
            )
        return value

    @model_validator(mode="after")
    def _distinct_centers(self) -> Self:
        if len({self.g_center, self.e_center, self.f_center}) != 3:
            raise ValueError("Classifier centers must be distinct")
        return self

    @field_serializer("g_center", "e_center", "f_center")
    def _encode_center(self, value: complex) -> ComplexTag:
        return encode_complex(value)


type ReadoutClassifier = Annotated[
    CircleClassifier | LinearClassifier | Centroid3Classifier,
    Field(discriminator="mode"),
]
"""Readout calibration selected by the required circle/linear/centroid3 mode."""

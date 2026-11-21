"""Inclusive endpoint bounds for sweep configuration annotations."""

from dataclasses import dataclass
from typing import Protocol

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class _SweepEndpoints(Protocol):
    @property
    def start(self) -> float: ...

    @property
    def stop(self) -> float: ...


@dataclass(frozen=True)
class SweepBounds:
    """Annotate a Pydantic sweep with inclusive endpoint limits in its units.

    minimum and maximum are the lower and upper bounds, respectively.
    The annotated type must expose numeric start and stop after validation.
    Either sweep direction is accepted. The first invalid endpoint fails
    validation, and valid sweep values are returned unchanged.
    """

    minimum: float
    maximum: float

    def endpoint_error(self, value: float) -> str | None:
        """Return a diagnostic if the evaluated endpoint is outside the bounds."""
        if self.minimum <= value <= self.maximum:
            return None
        return f"Sweep endpoint {value} must be in [{self.minimum:g}, {self.maximum:g}]"

    def _validate(self, sweep: _SweepEndpoints) -> _SweepEndpoints:
        for value in (sweep.start, sweep.stop):
            if (error := self.endpoint_error(value)) is not None:
                raise ValueError(error)
        return sweep

    def __get_pydantic_core_schema__(
        self, source_type: object, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Add endpoint validation after the wrapped sweep's Pydantic schema."""
        return core_schema.no_info_after_validator_function(
            self._validate, handler(source_type)
        )

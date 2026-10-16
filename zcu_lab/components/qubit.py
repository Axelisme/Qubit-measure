"""Shared qubit fields, channel wiring and predictor input conversion."""

from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from zcu_tools.resources.entry.schema import ComponentSchema
from zcu_tools.resources.entry.views import ComponentView, FieldView
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.predictor import PredictorInputError
from zcu_tools.simulate.transmon import TransmonParams

from .base import BuiltinComponent, Duration, Flux, Frequency, reject_null_channel


class QubitWiringSchema(BaseModel):
    """Optional drive and local flux channel indices; supplied values cannot be null."""

    model_config = ConfigDict(extra="forbid")

    ch: int | None = Field(
        default=None, ge=0, strict=True, description="Drive generator channel index."
    )
    lo_flux_ch: int | None = Field(
        default=None, ge=0, strict=True, description="Local flux channel index."
    )

    reject_null_channels = field_validator("ch", "lo_flux_ch", mode="before")(
        reject_null_channel
    )


class QubitBaseSchema(BuiltinComponent):
    """Fields shared by every qubit kind; physics energies belong to subclasses."""

    wiring: QubitWiringSchema = Field(
        default_factory=QubitWiringSchema,
        description="Drive and local flux channel wiring.",
    )
    freq: Frequency = Field(default=None, description="Qubit 0-1 transition frequency.")
    kappa: Frequency = Field(default=None, description="Qubit transition linewidth.")
    t1: Duration = Field(default=None, description="Energy relaxation time T1.")
    t2r: Duration = Field(default=None, description="Ramsey dephasing time T2*.")
    t2e: Duration = Field(default=None, description="Echo dephasing time T2E.")
    # These three values belong to global flux, interpreted by general.ext.flux_unit.
    flux_half: Flux = Field(
        default=None,
        description="Global flux at the half flux quantum, in general.ext.flux_unit.",
    )
    flux_period: Flux = Field(
        default=None,
        description="Global flux span of one flux quantum, in general.ext.flux_unit.",
    )
    flux_int: Flux = Field(
        default=None,
        description="Global flux at an integer flux quantum, in general.ext.flux_unit.",
    )
    flux_unit: Literal["A", "V"] | None = Field(
        default=None, description="Unit of the local flux bias: A or V."
    )
    resonator: str | None = Field(
        default=None, description="Name of the readout resonator component."
    )
    flux_source: str | None = Field(
        default=None, description="Name of the component that supplies flux bias."
    )


def params_for(
    component: ComponentSchema | ComponentView,
) -> FluxoniumParams | TransmonParams:
    """Read independent model inputs from a qubit component's current values.

    Accept a notebook Point/Setup ComponentView or an isolated ComponentSchema
    from a GUI context capture. Fluxonium reads EJ/EC/EL and SQUID transmon reads
    EJmax/EC/d; energies are GHz and d is dimensionless. Both read flux_half,
    flux_period and ext.flux_bias in native device units. Missing/unset fields
    stay None. No read refreshes, edits, adds defaults or builds a model.
    Unsupported kinds and nonnumeric/nonfinite supplied fields raise PredictorInputError
    naming the field.
    """
    if component.kind not in ("qubit/fluxonium", "qubit/transmon"):
        raise PredictorInputError(f"Unsupported predictor kind: {component.kind}")
    ext = getattr(component, "ext", None)
    if isinstance(ext, dict):
        bias = ext.get("flux_bias")
    elif isinstance(ext, FieldView):
        bias = getattr(ext, "flux_bias", None)
    else:
        bias = None

    def read(name: str) -> float | None:
        return _optional_number(getattr(component, name, None), name)

    affine = {
        "flux_half": read("flux_half"),
        "flux_period": read("flux_period"),
        "flux_bias": _optional_number(bias, "flux_bias"),
    }
    if component.kind == "qubit/transmon":
        return TransmonParams(EJmax=read("EJmax"), EC=read("EC"), d=read("d"), **affine)
    return FluxoniumParams(EJ=read("EJ"), EC=read("EC"), EL=read("EL"), **affine)


def _optional_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not isfinite(value)
    ):
        raise PredictorInputError(f"{name} must be a finite number or None")
    return float(value)

"""Typed zcu.t1-curve v1 files, separate from physical noise-fit validation."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from zcu_tools.resources._analysis_file import load_file, save_file
from zcu_tools.resources.analysis_source import AnalysisSource


class T1NoiseParams(BaseModel):
    """Partial T1 noise parameters or their uncertainty/initial values.

    Q_cap and Q_ind are dimensionless capacitive/inductive quality factors;
    x_qp is dimensionless quasiparticle density; Temp is temperature in K.
    Each value is finite or null. An omitted key is not an active parameter;
    an explicitly null key is active but unknown. No positivity, minimum
    channel count or temperature completeness is checked here.
    """

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    Q_cap: float | None = None
    x_qp: float | None = None
    Q_ind: float | None = None
    Temp: float | None = None


class T1CurveParams(BaseModel):
    """Partial noise-fit handoff and optimizer metadata, without sample arrays.

    params contains active noise values; null/absent means wholly unknown.
    stderr and init are partial uncertainties and starting values in the same
    units. fixed/free list active names, bounds maps active names to lower/upper
    endpoints (finite or null for unknown). init must use the same active noise
    names as params. cost/reduced_chi2 are nullable optimizer statistics; success
    is nullable optimizer status, not a storage completeness certificate.
    message/residual_mode/loss are nullable original optimizer text; max_nfev
    is a nullable integer evaluation limit. Omitted metadata stays omitted.
    """

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    params: T1NoiseParams | None = None
    stderr: T1NoiseParams | None = None
    init: T1NoiseParams | None = None
    fixed: tuple[str, ...] = ()
    free: tuple[str, ...] = ()
    bounds: dict[str, tuple[float | None, float | None]] = Field(default_factory=dict)
    cost: float | None = None
    reduced_chi2: float | None = None
    success: bool | None = None
    message: str | None = None
    residual_mode: str | None = None
    loss: str | None = None
    max_nfev: StrictInt | None = None

    @model_validator(mode="after")
    def active_names(self) -> Self:
        """Reject metadata naming absent channels; preserve unknown active values."""
        active = self.params.model_fields_set if self.params is not None else set()
        for key, names in (
            ("fixed", set(self.fixed)),
            ("free", set(self.free)),
            ("bounds", set(self.bounds)),
            (
                "stderr",
                self.stderr.model_fields_set if self.stderr is not None else set(),
            ),
            ("init", self.init.model_fields_set if self.init is not None else set()),
        ):
            if names - active:
                raise ValueError(
                    f"{key} contains inactive parameter names: {sorted(names - active)}"
                )
        if self.init is not None:
            noise = {"Q_cap", "x_qp", "Q_ind"}
            if self.init.model_fields_set & noise != active & noise:
                raise ValueError("init active noise names must match params")
        return self


class T1CurveFile(BaseModel):
    """A zcu.t1-curve v1 document.

    format identifies this artifact; version is strictly integer 1, not bool or
    float. params holds partial noise-fit data; source is required provenance.
    Extra envelope fields are rejected. This is not an analysis completeness
    certificate. Callers must supply all four envelope fields.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    format: Literal["zcu.t1-curve"]
    version: StrictInt = Field(ge=1, le=1)
    params: T1CurveParams
    source: AnalysisSource


def load_t1_curve_file(path: Path) -> T1CurveFile:
    """Read exact path as zcu.t1-curve v1 without changing it.

    Invalid JSON/schema raises ValueError; missing files raise FileNotFoundError
    and other I/O failures propagate. No time, run ID, absent channel values or
    physical completeness is inferred.
    """
    return load_file(path, T1CurveFile)


def save_t1_curve_file(path: Path, document: T1CurveFile) -> None:
    """Validate and atomically replace exact path; its parent must exist.

    Preserve nulls, omitted fields and provenance strings; always write the fixed
    header. Invalid input raises ValueError before I/O. I/O failures propagate,
    retaining the old file before replacement and cleaning sibling temporary
    files. No cross-file transaction, writer lock or power-loss guarantee.
    """
    save_file(path, document, T1CurveFile)

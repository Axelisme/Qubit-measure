"""Typed zcu.dispersive v1 files; storage validation is not a physical fit check."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt

from ._analysis_file import load_file, save_file
from .analysis_source import AnalysisSource


class PlotTransitions(BaseModel):
    """Resonator seed and opaque authored transition annotations.

    r_f is a finite resonator frequency in GHz or null when unknown. Remaining
    string keys contain original JSON annotations; storage does not interpret
    transition names, infer units or synthesize line data.
    """

    model_config = ConfigDict(strict=True, extra="allow", allow_inf_nan=False)
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    r_f: float | None = None


class DispersiveParams(BaseModel):
    """Partial dispersive fit data, with no positivity/completeness requirement.

    g is charge-mode coupling in GHz; bare_rf is bare resonator frequency in
    GHz. Either can be null or absent when unknown. plot_transitions preserves
    the separately authored resonator seed and transition annotations.
    Finite numbers only; booleans and numeric strings are rejected.
    """

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    g: float | None = None
    bare_rf: float | None = None
    plot_transitions: PlotTransitions = Field(default_factory=PlotTransitions)


class DispersiveFile(BaseModel):
    """A zcu.dispersive v1 document.

    format identifies this artifact; version is strictly integer 1, not bool or
    float. params contains partial fit values; source is required provenance.
    No extra envelope fields are accepted. This represents unknowns, not an
    analysis-ready fit. Callers must supply all four envelope fields.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    format: Literal["zcu.dispersive"]
    version: StrictInt = Field(ge=1, le=1)
    params: DispersiveParams
    source: AnalysisSource


def load_dispersive_file(path: Path) -> DispersiveFile:
    """Read exact path as zcu.dispersive v1 without changing it.

    Raise ValueError for invalid JSON/schema, FileNotFoundError for missing
    files; other I/O errors propagate. Unknown values stay null/absent and no
    source metadata or physical completeness is inferred.
    """
    return load_file(path, DispersiveFile)


def save_dispersive_file(path: Path, document: DispersiveFile) -> None:
    """Validate and atomically replace exact path; its parent must exist.

    Preserve explicit nulls, omitted content fields, transition annotations and
    source strings. Fixed format/version are always written. ValueError rejects
    invalid input before I/O. I/O failures propagate, retaining the old file
    before replacement and cleaning the sibling temporary. No multiwriter lock,
    cross-file transaction or power-loss durability is provided.
    """
    save_file(path, document, DispersiveFile)

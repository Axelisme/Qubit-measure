"""Six independent fluxonium settings and safe YAML loading; no model creation."""

from pathlib import Path

from pydantic import ValidationError

from zcu_tools.resources.simulation_schema import validation_message
from zcu_tools.resources.yaml import load_yaml

from .schema import SixFluxoniumSettings

__all__ = ["SixFluxoniumSettings", "load_six_fluxonium", "parse_six_fluxonium"]


def parse_six_fluxonium(document: object) -> SixFluxoniumSettings:
    """Validate decoded six-fluxonium v1 material without I/O or device creation.

    Accept a mapping or SixFluxoniumSettings. Require Q1..Q6 physical materials,
    one global DC source and the fixed channel map on SixFluxoniumSettings.
    Unknown fields, invalid physical values and invalid wiring raise ValueError
    with dotted field diagnostics. The returned snapshot owns its nested values.
    """
    try:
        return SixFluxoniumSettings.model_validate(document)
    except ValidationError as error:
        raise ValueError(validation_message(error, root="model")) from error


def load_six_fluxonium(path: str | Path) -> SixFluxoniumSettings:
    """Read UTF-8 YAML at path (relative to cwd) and validate complete material.

    OSError propagates. Invalid YAML, repeated keys and invalid material raise
    ValueError with the path and YAML location or dotted field diagnostics.
    No files are written and no devices or numerical models are created.
    """
    path = Path(path)
    try:
        return parse_six_fluxonium(load_yaml(path))
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error

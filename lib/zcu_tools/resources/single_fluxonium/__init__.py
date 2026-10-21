"""Single-fluxonium YAML owner; no device creation or numeric simulation."""

from pathlib import Path

from pydantic import ValidationError

from zcu_tools.resources.simulation_schema import validation_message
from zcu_tools.resources.yaml import load_yaml

from .schema import SingleFluxoniumSettings


def parse_single_fluxonium(document: object) -> SingleFluxoniumSettings:
    """Validate a decoded external document into an immutable named snapshot.

    Accept a mapping or SingleFluxoniumSettings with the schema's required
    sections and units. Unknown fields, invalid dimensions/units, nonfinite
    numbers and colliding source names/generators raise ValueError with dotted
    field paths. No I/O, mutation, device creation or convergence check occurs.
    """
    try:
        return SingleFluxoniumSettings.model_validate(document)
    except ValidationError as error:
        raise ValueError(validation_message(error, root="model")) from error


def load_single_fluxonium(path: str | Path) -> SingleFluxoniumSettings:
    """Read one UTF-8 physical settings YAML into an immutable snapshot.

    Relative paths use the caller's cwd. OSError propagates. Invalid YAML,
    repeated explicit keys or schema/wiring violations raise ValueError with
    the file path and YAML location or dotted field diagnostics. YAML merges
    may supply defaults overridden by explicit keys. No files are written and
    no resources are created. This does not load zcu.entry-template documents;
    those use resources.entry.load_entry_template.
    """
    path = Path(path)
    document = load_yaml(path)
    try:
        return parse_single_fluxonium(document)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error

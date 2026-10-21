"""FTF YAML owner: parse once into named values with dotted field diagnostics.

No devices, program compilation or numerical model is constructed here. Import
load_ftf for files or parse_ftf for in-memory documents. Schema defaults and units
are documented on the named value types in schema.py.
"""

from pathlib import Path

from pydantic import ValidationError

from zcu_tools.resources.simulation_schema import validation_message
from zcu_tools.resources.yaml import load_yaml

from .schema import FTFSettings


def _wiring(config: FTFSettings) -> None:
    names = set(config.dc_bias.sources)
    channels = {config.readout.drive.gen_ch}
    mux = config.readout.mux
    if mux is not None:
        if mux.gen_ch in channels:
            raise ValueError("readout.mux.gen_ch: duplicate generator binding")
        channels.add(mux.gen_ch)
    for name, control in config.drive_lines.items():
        if control.gen_ch in channels:
            raise ValueError(f"drive_lines.{name}.gen_ch: duplicate generator binding")
        if name in names:
            raise ValueError(f"drive_lines.{name}: duplicate source name")
        names.add(name)
        channels.add(control.gen_ch)
    pair = config.readout.resonators
    if not any(r.kappa_external_mhz > 0 for r in (pair.r1, pair.r2)):
        raise ValueError("readout.resonators: shared line must have external coupling")


def parse_ftf(document: object) -> FTFSettings:
    """Validate a decoded YAML document and return an immutable named snapshot.

    document is an external value, not a numeric kernel parameter. Unknown fields,
    dimensions, units, duplicate sources/generators, source limits and receiver PSD
    errors raise ValueError with a dotted field path. No I/O or resource creation.
    """
    try:
        config = FTFSettings.model_validate(document)
    except ValidationError as error:
        raise ValueError(validation_message(error, root="")) from error
    _wiring(config)
    return config


def load_ftf(path: str | Path) -> FTFSettings:
    """Read one UTF-8 YAML file into an immutable validated FTF snapshot.

    Relative paths use the caller's cwd. OSError propagates. YAML syntax and
    schema/wiring errors raise ValueError including path and field diagnostics.
    Repeated explicit YAML keys raise ValueError with the key and YAML location;
    merge mappings may supply defaults overridden by explicit keys.
    This function never creates devices or compiles a program.
    """
    path = Path(path)
    document = load_yaml(path)
    try:
        return parse_ftf(document)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error

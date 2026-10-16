"""FTF YAML owner: parse once into named values with dotted field diagnostics.

No devices, program compilation or numerical model is constructed here. Import
load_ftf for files or parse_ftf for in-memory documents. Schema defaults and units
are documented on the named value types in schema.py.
"""

from pathlib import Path

import yaml
from pydantic import ValidationError

from .schema import FTFSettings


def _dc_wiring(config: FTFSettings) -> set[str]:
    names: set[str] = set()
    sources = {}
    for i, source in enumerate(config.devices):
        if source.name in names:
            raise ValueError(f"devices.{i}.name: duplicate device binding")
        names.add(source.name)
        sources[source.name] = source
    columns: set[str] = set()
    for i, column in enumerate(config.dc_mapping.columns):
        path = f"dc_mapping.columns.{i}"
        if column.source not in sources:
            raise ValueError(f"{path}.source: undefined DC source")
        if column.source in columns:
            raise ValueError(f"{path}.source: duplicate DC column")
        columns.add(column.source)
        if column.unit != f"Phi0/{sources[column.source].unit}":
            raise ValueError(f"{path}.unit: calibration differs from device unit")
    return names


def _wiring(config: FTFSettings) -> None:
    names = _dc_wiring(config)
    channels = {config.readout_source.gen_ch}
    if not 0 <= config.readout_source.gen_ch < config.channels.generators:
        raise ValueError("readout_source.gen_ch: exceeds generator count")
    for family, controls in (
        ("flux_sources", config.flux_sources),
        ("microwave_sources", config.microwave_sources),
    ):
        for i, control in enumerate(controls):
            if (
                control.gen_ch in channels
                or control.gen_ch >= config.channels.generators
            ):
                raise ValueError(
                    f"{family}.{i}.gen_ch: duplicate or out-of-range generator binding"
                )
            if control.name in names:
                raise ValueError(f"{family}.{i}.name: duplicate source name")
            names.add(control.name)
            channels.add(control.gen_ch)
    if config.receiver.ro_ch >= config.channels.readouts:
        raise ValueError("receiver.ro_ch: exceeds readout count")
    if not any(r.kappa_external_mhz > 0 for r in config.resonators):
        raise ValueError("resonators: shared line must have external coupling")


def parse_ftf(document: object) -> FTFSettings:
    """Validate a decoded YAML document and return an immutable named snapshot.

    document is an external value, not a numeric kernel parameter. Unknown fields,
    dimensions, units, duplicate/unbound sources, source limits and receiver PSD
    errors raise ValueError with a dotted field path. No I/O or resource creation.
    """
    try:
        config = FTFSettings.model_validate(document)
    except ValidationError as error:
        messages = []
        for item in error.errors():
            path = ".".join(str(part) for part in item["loc"])
            message = item["msg"].removeprefix("Value error, ")
            if ": " in message:
                field, message = message.split(": ", 1)
                path = f"{path}.{field}" if path else field
            messages.append(f"{path}: {message}")
        raise ValueError("; ".join(messages)) from error
    _wiring(config)
    return config


def load_ftf(path: str | Path) -> FTFSettings:
    """Read one UTF-8 YAML file into an immutable validated FTF snapshot.

    Relative paths use the caller's cwd. OSError propagates. YAML syntax and
    schema/wiring errors raise ValueError including path and field diagnostics.
    This function never creates devices or compiles a program.
    """
    path = Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: invalid YAML: {error}") from error
    try:
        return parse_ftf(document)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error

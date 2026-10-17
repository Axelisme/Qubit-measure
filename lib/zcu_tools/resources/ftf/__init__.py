"""FTF YAML owner: parse once into named values with dotted field diagnostics.

No devices, program compilation or numerical model is constructed here. Import
load_ftf for files or parse_ftf for in-memory documents. Schema defaults and units
are documented on the named value types in schema.py.
"""

from collections.abc import Hashable
from pathlib import Path

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, ScalarNode

from .schema import FTFSettings


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject repeated explicit keys before PyYAML expands merge mappings."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._checked_mappings: set[MappingNode] = set()

    def flatten_mapping(self, node: MappingNode) -> None:
        # Aliases can revisit nodes after flattening; merged overrides are valid.
        if node not in self._checked_mappings:
            keys: set[Hashable] = set()
            for key_node, _ in node.value:
                if isinstance(key_node, ScalarNode) and key_node.tag in (
                    "tag:yaml.org,2002:merge",
                    "tag:yaml.org,2002:value",
                ):
                    key = key_node.value
                else:
                    key = self.construct_object(key_node)
                if not isinstance(key, Hashable):
                    raise ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "found unhashable key",
                        key_node.start_mark,
                    )
                if key in keys:
                    raise ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"duplicate key {key!r}",
                        key_node.start_mark,
                    )
                keys.add(key)
            self._checked_mappings.add(node)
        super().flatten_mapping(node)


def _wiring(config: FTFSettings) -> None:
    names = set(config.dc_bias.sources)
    channels = {config.readout.drive.gen_ch}
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
    Repeated explicit YAML keys raise ValueError with the key and YAML location;
    merge mappings may supply defaults overridden by explicit keys.
    This function never creates devices or compiles a program.
    """
    path = Path(path)
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: invalid YAML: {error}") from error
    try:
        return parse_ftf(document)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error

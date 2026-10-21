"""Safe YAML decoding shared by experiment settings loaders."""

from collections.abc import Hashable
from pathlib import Path

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, ScalarNode


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


def load_yaml(path: str | Path) -> object:
    """Decode one UTF-8 YAML file without interpreting its domain schema.

    Relative paths use the caller\'s cwd. Return the decoded document, including
    None for an empty file. No writes occur. OSError propagates. Invalid YAML
    or repeated explicit mapping keys raise ValueError with file and YAML
    location diagnostics. Safe YAML tags only; merge defaults and explicit
    overrides are supported. Domain owners validate the returned external value.
    """
    path = Path(path)
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: invalid YAML: {error}") from error

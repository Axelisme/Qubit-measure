"""Caller-owned component schema versions, separate from YAML container versions."""

from dataclasses import dataclass
from pathlib import Path
from shlex import join

from zcu_tools.format_version import YamlMap


@dataclass(frozen=True)
class ParameterSchemaPolicy:
    """One lab's current whole-document component version and upgrade command.

    current_version is a positive integer, not a container format_version.
    upgrade_command is a nonempty argv prefix for the caller's durable upgrade
    entrypoint; the failing YAML path is appended as one argument in diagnostics.
    Register this policy alongside component models, never infer it from kinds.
    Invalid versions or empty command arguments raise ValueError at construction.
    This record keeps version and remediation consistent across entry readers,
    templates, creation and the offline runner without teaching lib lab shapes.
    """

    current_version: int
    upgrade_command: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.current_version) is not int or self.current_version < 1:
            raise ValueError("current_version must be a positive integer")
        if not self.upgrade_command or any(not arg for arg in self.upgrade_command):
            raise ValueError("upgrade_command must contain nonempty argv arguments")

    def require_current(self, document: YamlMap, *, source: Path) -> None:
        """Reject absent, malformed or noncurrent schema_version without I/O.

        document is a decoded complete YAML mapping; source locates diagnostics.
        ValueError includes stored/current versions and a quoted dry-run command.
        Only a strict integer equal to current_version succeeds. No legacy
        classification or component validation occurs here.
        """
        version = document.get("schema_version")
        if type(version) is int and version == self.current_version:
            return
        stored = repr(version) if "schema_version" in document else "missing"
        command = join((*self.upgrade_command, str(source)))
        raise ValueError(
            f"{source}: schema_version {stored}; current {self.current_version}; "
            f"run upgrade: {command}"
        )

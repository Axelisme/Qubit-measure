"""Neutral typed declaration passed to the offline CLI composition root."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from zcu_tools.experiment.metadata import CfgSnapshot

from migrations.legacy_labber import LegacyVariableSchema


@dataclass(frozen=True)
class MigrationExperiment:
    """Bind a concrete typed reader to its generic schema without erasing its cfg.

    source_tag and cfg_type are the exact historical experiment/cfg identities;
    neither determines the other. native_tag is the canonical spec tag written
    into native RunMetadata; it may differ from source_tag only by an explicit
    migration declaration. schemas is the generic
    disk-unit variable declaration. validate_cfg checks a historical CfgSnapshot
    for storage format, this pair's cfg_type, major version and concrete model.
    It returns None without changing raw values and raises ValueError for
    unconvertible cfg.
    validate_native reads an exact native Path
    with the corresponding typed load_run, returns None and propagates failures.
    These bindings perform no acquisition, registration or live-context capture.
    """

    source_tag: str
    native_tag: str
    cfg_type: str
    schemas: tuple[LegacyVariableSchema, ...]
    validate_cfg: Callable[[CfgSnapshot], None]
    validate_native: Callable[[Path], None]

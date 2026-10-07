"""Offline legacy conversion inputs, evidence loading and durable state models.

This owner never imports experiment or concrete lab definitions. Normal runtime
loaders do not use legacy fallback. The shipped CLI supplies explicit mapping,
entry definition registration and native experiment validation.
"""

from .converter import migrate_storage
from .errors import MigrationInputError
from .evidence import load_run_evidence
from .models import (
    FailureItem,
    FileMigrationItem,
    KeyMappingItem,
    KeyRule,
    LegacyRunEvidence,
    LegacySnapshotEvidence,
    MappingAction,
    MigrationDestination,
    MigrationFileState,
    MigrationIdentity,
    MigrationManifest,
    MigrationMapping,
    MigrationPart,
    MigrationReport,
    MigrationRequest,
    MigrationRunEvidenceDocument,
    MigrationSource,
    ModuleRule,
    PendingItem,
    RunAssignment,
)
from .state import report_json

__all__ = [
    "FailureItem",
    "FileMigrationItem",
    "KeyMappingItem",
    "KeyRule",
    "LegacyRunEvidence",
    "LegacySnapshotEvidence",
    "MappingAction",
    "MigrationDestination",
    "MigrationFileState",
    "MigrationIdentity",
    "MigrationInputError",
    "MigrationManifest",
    "MigrationMapping",
    "MigrationPart",
    "MigrationReport",
    "MigrationRequest",
    "MigrationRunEvidenceDocument",
    "MigrationSource",
    "ModuleRule",
    "PendingItem",
    "RunAssignment",
    "load_run_evidence",
    "migrate_storage",
    "report_json",
]

"""Explicit-root parameter containers, independent of legacy context services."""

from ._path_segments import validate_path_segment
from .errors import (
    DestinationExistsError,
    MissingPointError,
    MissingReferenceError,
    RoleResolutionError,
    UnknownFieldError,
    UnknownKindError,
)
from .ledger import LedgerEntry, RecordsLedger
from .ledger_models import (
    AcceptedPayload,
    AcceptedWrite,
    AcquiredPayload,
    AnalyzedPayload,
    ImportPayload,
    JsonObject,
    JsonValue,
    LedgerAcquisition,
    LedgerEvent,
    LedgerEventContent,
    LedgerSavedPaths,
    Origin,
    OutputFormat,
    SavedOutput,
    SavedPayload,
    SourceReference,
)
from .points import PointView
from .registry import ComponentRegistry, RoleRegistry, RoleSpec, component_registry
from .result_entry import (
    EntryIdentity,
    ResultEntry,
    entry_component_paths,
    entry_predictor_paths,
    rename_entry,
    validate_point_creation,
)
from .roles import RoleView, role_registry
from .save_layout import (
    ArtifactKey,
    Output,
    SaveLayout,
    WorkflowRoots,
    new_run_id,
    workflow_roots,
)
from .save_names import SaveContext, default_save_stem
from .schema import ComponentSchema, ParameterContent
from .schema_fields import (
    FieldSpec,
    FieldValueKind,
    component_field_specs,
    role_candidates,
)
from .storage import (
    RunOutputBinding,
    StorageBinding,
    bind_run_outputs,
    capture_entry_storage,
    capture_point_storage,
)
from .template import EntryTemplate, load_entry_template
from .views import (
    ComponentView,
    DocumentField,
    EditView,
    SetupView,
    document_fields,
    document_meta,
    document_revision,
    document_unit,
    document_value,
    parameter_content,
    resolve_path,
    snapshot,
    source_path,
)
from .writeback import (
    CommittedWrite,
    EntryWritebackTarget,
    ModuleWriteback,
    ParameterWriteback,
    PointWritebackTarget,
    RecordWriteback,
    WaveformWriteback,
    WritebackApplyError,
    WritebackEvidence,
    WritebackItem,
    WritebackResult,
    apply_writeback,
)

"""Public contracts for owner-sequenced, versioned cfg resources.

Editing handles identify one resource for its entire lifetime. Implementations
must detach inputs and observations, publish batches atomically, and reject
mutations and acceptance during notification. App composition owns creation,
revocation, and the allocation of the narrower editing/acceptance capabilities.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol, TypeAlias
from uuid import uuid4

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

from .binding.draft import CfgDraft
from .binding.observation import CfgNodeObservation
from .binding.ports import ExpressionEvaluator, OptionProvider, ReferenceCatalog
from .lowering import RangeFactory
from .model import (
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    DirectValue,
    EvalValue,
    ReferenceSpec,
)
from .resolved import lower_resolved_cfg

CfgPath: TypeAlias = tuple[str, ...]
CfgInput: TypeAlias = (
    None
    | bool
    | int
    | float
    | complex
    | str
    | DirectValue
    | EvalValue
    | Sequence["CfgInput"]
    | Mapping[str, "CfgInput"]
)


class CfgInputReason(str, Enum):
    MALFORMED_INPUT = "malformed_input"
    UNKNOWN_PATH = "unknown_path"
    READONLY = "readonly"
    UNSUPPORTED_MODE = "unsupported_mode"
    INVALID_VALUE = "invalid_value"
    CAPTURE_SYNTAX = "capture_syntax"


class CfgPreconditionReason(str, Enum):
    STALE_REVISION = "stale_revision"
    RESOURCE_GONE = "resource_gone"
    MUTATION_BLOCKED = "mutation_blocked"
    REENTRANT_MUTATION = "reentrant_mutation"
    CAPTURE_UNAVAILABLE = "capture_unavailable"
    NOT_VALID = "not_valid"


class CfgInputError(InvalidInputError):
    def __init__(
        self,
        reason: CfgInputReason,
        message: str,
        *,
        path: CfgPath | None = None,
        edit_index: int | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason.value)
        self.reason = reason
        self.path = path
        self.edit_index = edit_index


class CfgPreconditionError(FailedPreconditionError):
    def __init__(
        self,
        reason: CfgPreconditionReason,
        message: str,
        *,
        path: CfgPath | None = None,
        edit_index: int | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason.value)
        self.reason = reason
        self.path = path
        self.edit_index = edit_index


class CfgId(str):
    def __new__(cls, value: object) -> CfgId:
        if not isinstance(value, str) or not value:
            raise CfgInputError(
                CfgInputReason.MALFORMED_INPUT, "cfg_id must be a nonempty string"
            )
        return super().__new__(cls, value)


class CfgRevision(int):
    def __new__(cls, value: object) -> CfgRevision:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CfgInputError(
                CfgInputReason.MALFORMED_INPUT,
                "revision must be a nonnegative integer, not bool",
            )
        return super().__new__(cls, value)


@dataclass(frozen=True)
class CfgRef:
    cfg_id: CfgId
    revision: CfgRevision


@dataclass(frozen=True)
class CfgEdit:
    path: CfgPath
    value: CfgInput

    def __post_init__(self) -> None:
        _validate_path(self.path)


class CfgStatus(str, Enum):
    VALID = "Valid"
    INVALID = "Invalid"
    UNAVAILABLE = "Unavailable"


@dataclass(frozen=True)
class SourceRevision:
    source_id: str
    revision: CfgRevision


SourceBasis: TypeAlias = tuple[SourceRevision, ...]


@dataclass(frozen=True)
class CfgDiagnostic:
    path: CfgPath
    reason: str
    message: str


@dataclass(frozen=True)
class CfgObservation:
    ref: CfgRef
    status: CfgStatus
    tree: CfgNodeObservation
    source_basis: SourceBasis
    diagnostics: tuple[CfgDiagnostic, ...]


@dataclass(frozen=True)
class AcceptedConfig:
    ref: CfgRef
    source_basis: SourceBasis
    values: dict[str, object]


class CfgStaleError(CfgPreconditionError):
    def __init__(self, expected: CfgRef, actual: CfgRef) -> None:
        super().__init__(
            CfgPreconditionReason.STALE_REVISION,
            f"Expected cfg revision {expected.revision}, current is {actual.revision}",
        )
        self.expected = expected
        self.actual = actual


class CfgEditing(Protocol):
    def observe(self) -> CfgObservation: ...

    def watch(
        self, callback: Callable[[CfgObservation], None]
    ) -> Callable[[], None]: ...

    def edit(
        self, expected_revision: CfgRevision, edits: tuple[CfgEdit, ...]
    ) -> CfgObservation: ...

    def reset(self, expected_revision: CfgRevision) -> CfgObservation: ...

    def refresh(self, expected_revision: CfgRevision) -> CfgObservation: ...


class CfgAcceptance(Protocol):
    def accept(self, expected_revision: CfgRevision) -> AcceptedConfig: ...


@dataclass(frozen=True)
class CfgResolution:
    """One command's stable, already-published source view (never live I/O).

    Composition binds these ports to detached source snapshots. A new view is
    acquired for a command, not for observe, watch, or acceptance.
    """

    source_basis: SourceBasis
    evaluate_expression: ExpressionEvaluator
    provide_options: OptionProvider
    references: ReferenceCatalog
    read_capture: Callable[[str], object]
    validate_expression: Callable[[str], None]


class CfgResource:
    """Single owner for a private candidate tree and detached publications.

    Only app composition receives this concrete owner. Frontends receive its
    CfgEditing capability; run owners receive CfgAcceptance. All calls are
    synchronous on the same owner sequence. ``defaults`` executes only at
    creation/reset and must return the same definition throughout this lifetime.
    """

    def __init__(
        self,
        defaults: Callable[[], CfgSchema],
        *,
        resolution: Callable[[], CfgResolution],
        make_range: RangeFactory,
        mutation_allowed: Callable[[], bool] = lambda: True,
    ) -> None:
        self._defaults = defaults
        self._resolution = resolution
        self._make_range = make_range
        self._mutation_allowed = mutation_allowed
        self._closed = False
        self._notifying = 0
        self._subscribers: dict[object, Callable[[CfgObservation], None]] = {}
        schema = deepcopy(defaults())
        _validate_definition(schema.spec)
        self._spec = deepcopy(schema.spec)
        self._draft, basis = self._prepare(schema)
        self._observation = self._observe_draft(
            self._draft, CfgRef(CfgId(uuid4().hex), CfgRevision(0)), basis
        )

    def observe(self) -> CfgObservation:
        self._require_open()
        return deepcopy(self._observation)

    def watch(self, callback: Callable[[CfgObservation], None]) -> Callable[[], None]:
        self._require_open()
        token = object()
        self._subscribers[token] = callback
        self._deliver(callback)

        def unsubscribe() -> None:
            self._subscribers.pop(token, None)

        return unsubscribe

    def edit(
        self, expected_revision: CfgRevision, edits: tuple[CfgEdit, ...]
    ) -> CfgObservation:
        from ._capture import ExpressionCapture
        from ._node_write import write_node

        edits = _validate_edits(edits)
        self._check_command(expected_revision)
        source = self._resolution()
        capture = ExpressionCapture(source.read_capture, source.validate_expression)
        failures: list[Exception] = []
        candidate, basis = self._prepare(
            deepcopy(self._draft.snapshot()), source=source, failures=failures
        )
        try:
            for index, edit in enumerate(edits):
                try:
                    write_node(
                        candidate.root, edit.path, deepcopy(edit.value), capture.prepare
                    )
                except CfgInputError as exc:
                    raise CfgInputError(
                        exc.reason, str(exc), path=edit.path, edit_index=index
                    ) from exc
                except CfgPreconditionError as exc:
                    raise CfgPreconditionError(
                        exc.reason, str(exc), path=edit.path, edit_index=index
                    ) from exc
            if failures:
                raise failures[0]
            observation = self._next_observation(candidate, basis)
        except BaseException:
            candidate.close()
            raise
        return self._publish(candidate, observation)

    def reset(self, expected_revision: CfgRevision) -> CfgObservation:
        self._check_command(expected_revision)
        schema = deepcopy(self._defaults())
        if schema.spec != self._spec:
            raise RuntimeError("defaults changed the resource definition")
        candidate, basis = self._prepare(schema)
        try:
            observation = self._next_observation(candidate, basis)
        except BaseException:
            candidate.close()
            raise
        return self._publish(candidate, observation)

    def refresh(self, expected_revision: CfgRevision) -> CfgObservation:
        self._check_command(expected_revision)
        source: CfgResolution | None = None
        candidate: CfgDraft | None = None
        try:
            source = self._resolution()
            candidate, basis = self._prepare(
                deepcopy(self._draft.snapshot()), source=source
            )
            observation = self._next_observation(candidate, basis)
        except Exception as exc:
            if candidate is not None:
                candidate.close()
            logging.getLogger(__name__).exception(
                "Cfg refresh failed; publishing Unavailable"
            )
            return self._publish_unavailable(exc, source)
        return self._publish(candidate, observation)

    def accept(self, expected_revision: CfgRevision) -> AcceptedConfig:
        self._check_command(expected_revision, editing=False)
        if self._observation.status is not CfgStatus.VALID:
            raise CfgPreconditionError(
                CfgPreconditionReason.NOT_VALID, "cfg is not valid"
            )
        values = lower_resolved_cfg(self._draft.snapshot(), make_range=self._make_range)
        return AcceptedConfig(
            self._observation.ref, self._observation.source_basis, deepcopy(values)
        )

    def revoke(self) -> None:
        if self._closed:
            return
        self._require_not_notifying()
        self._closed = True
        self._subscribers.clear()
        self._draft.close()

    def _require_open(self) -> None:
        if self._closed:
            raise CfgPreconditionError(
                CfgPreconditionReason.RESOURCE_GONE, "cfg resource is revoked"
            )

    def _require_not_notifying(self) -> None:
        if self._notifying:
            raise CfgPreconditionError(
                CfgPreconditionReason.REENTRANT_MUTATION,
                "cfg mutation and acceptance are forbidden during notification",
            )

    def _check_command(self, expected: CfgRevision, *, editing: bool = True) -> None:
        self._require_open()
        expected = CfgRevision(expected)
        actual = self._observation.ref
        if expected != actual.revision:
            raise CfgStaleError(CfgRef(actual.cfg_id, expected), actual)
        self._require_not_notifying()
        if editing and not self._mutation_allowed():
            raise CfgPreconditionError(
                CfgPreconditionReason.MUTATION_BLOCKED, "cfg editing is blocked"
            )

    def _prepare(
        self,
        schema: CfgSchema,
        *,
        source: CfgResolution | None = None,
        failures: list[Exception] | None = None,
    ) -> tuple[CfgDraft, SourceBasis]:
        source = self._resolution() if source is None else source
        failures = [] if failures is None else failures

        def evaluate(expression: str) -> int | float | complex:
            try:
                return source.evaluate_expression(expression)
            except InvalidInputError:
                raise
            except Exception as exc:
                # Binding caches display errors. Preserve unexpected failures for
                # the command owner instead of publishing them as invalid input.
                failures.append(exc)
                raise

        draft = CfgDraft(
            schema,
            evaluate_expression=evaluate,
            provide_options=source.provide_options,
            references=source.references,
        )
        if failures:
            draft.close()
            raise failures[0]
        return draft, source.source_basis

    def _next_observation(self, draft: CfgDraft, basis: SourceBasis) -> CfgObservation:
        ref = replace(
            self._observation.ref,
            revision=CfgRevision(self._observation.ref.revision + 1),
        )
        return self._observe_draft(draft, ref, basis)

    @staticmethod
    def _observe_draft(
        draft: CfgDraft, ref: CfgRef, basis: SourceBasis
    ) -> CfgObservation:
        tree = draft.observe()
        diagnostics = tuple(_diagnostics(tree))
        status = CfgStatus.VALID if tree.valid else CfgStatus.INVALID
        return CfgObservation(ref, status, tree, basis, diagnostics)

    def _publish(
        self, candidate: CfgDraft, observation: CfgObservation
    ) -> CfgObservation:
        previous = self._draft
        self._draft = candidate
        self._observation = observation
        previous.close()
        self._notify()
        return deepcopy(observation)

    def _publish_unavailable(
        self, failure: Exception, source: CfgResolution | None
    ) -> CfgObservation:
        from ._unavailable import unavailable_tree

        observation = CfgObservation(
            replace(
                self._observation.ref,
                revision=CfgRevision(self._observation.ref.revision + 1),
            ),
            CfgStatus.UNAVAILABLE,
            unavailable_tree(self._draft.observe()),
            () if source is None else source.source_basis,
            (
                CfgDiagnostic(
                    (),
                    "source_failure" if source is None else "resolution_failure",
                    str(failure),
                ),
            ),
        )
        self._observation = observation
        self._notify()
        return deepcopy(observation)

    def _notify(self) -> None:
        for token, callback in tuple(self._subscribers.items()):
            if token in self._subscribers:
                self._deliver(callback)

    def _deliver(self, callback: Callable[[CfgObservation], None]) -> None:
        self._notifying += 1
        try:
            callback(deepcopy(self._observation))
        except Exception:
            logging.getLogger(__name__).exception(
                "Cfg subscriber failed after publication"
            )
        finally:
            self._notifying -= 1


def _validate_path(path: object) -> None:
    if not isinstance(path, tuple) or any(
        not isinstance(part, str) or not part for part in path
    ):
        raise CfgInputError(
            CfgInputReason.MALFORMED_INPUT,
            "path must be a tuple of nonempty string segments",
        )


def _validate_edits(edits: object) -> tuple[CfgEdit, ...]:
    if not isinstance(edits, tuple) or any(
        not isinstance(edit, CfgEdit) for edit in edits
    ):
        raise CfgInputError(
            CfgInputReason.MALFORMED_INPUT, "edits must be a tuple of CfgEdit"
        )
    return edits


def _validate_field_name(key: object) -> None:
    if not isinstance(key, str) or not key or key.startswith("__"):
        raise ValueError(
            f"Cfg definition has an invalid or reserved field name: {key!r}"
        )


def _validate_definition(spec: CfgNodeSpec) -> None:
    if isinstance(spec, CfgSectionSpec):
        for key, child in spec.fields.items():
            _validate_field_name(key)
            _validate_definition(child)
    elif isinstance(spec, ReferenceSpec):
        for shape in spec.allowed:
            _validate_definition(shape)


def _diagnostics(tree: CfgNodeObservation, path: CfgPath = ()):
    if not tree.valid and not tree.children:
        value = tree.value
        message = "input is incomplete or invalid"
        if isinstance(value, (DirectValue, EvalValue)):
            message = value.error or value.validation_error or message
        yield CfgDiagnostic(path, "invalid_input", message)
    for key, child in tree.children.items():
        yield from _diagnostics(child, (*path, key))

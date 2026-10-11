"""Public contracts for owner-sequenced cfg resources.

Editing handles identify one resource for its entire lifetime. Implementations
must detach inputs and observations, publish batches atomically, and reject
mutations and acceptance during notification. App composition owns creation,
revocation, and the allocation of the narrower editing/acceptance capabilities.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

from .binding.draft import CfgDraft
from .binding.observation import CfgNodeObservation
from .binding.ports import ExpressionEvaluator, OptionProvider, ReferenceCatalog
from .lowering import LoweredData
from .model import (
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    DirectValue,
    EvalValue,
    ReferenceSpec,
    ReferenceValue,
    SweepValue,
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
    revision: int


SourceBasis: TypeAlias = tuple[SourceRevision, ...]


@dataclass(frozen=True)
class CfgDiagnostic:
    path: CfgPath
    reason: str
    message: str


@dataclass(frozen=True)
class CfgObservation:
    status: CfgStatus
    tree: CfgNodeObservation
    source_basis: SourceBasis
    diagnostics: tuple[CfgDiagnostic, ...]


@dataclass(frozen=True)
class AcceptedConfig:
    source_basis: SourceBasis
    values: dict[str, LoweredData]


class CfgEditing(Protocol):
    def observe(self) -> CfgObservation: ...

    def watch(
        self, callback: Callable[[CfgObservation], None]
    ) -> Callable[[], None]: ...

    def edit(self, edits: tuple[CfgEdit, ...]) -> CfgObservation: ...

    def select_custom_reference(self, path: CfgPath, label: str) -> CfgObservation: ...

    def reset(self) -> CfgObservation: ...

    def refresh(self) -> CfgObservation: ...


class CfgAcceptance(Protocol):
    def accept(
        self,
        *,
        resolution: CfgResolution | None = None,
    ) -> AcceptedConfig:
        """Accept the current inputs, optionally against a fixed one-off source.

        Without ``resolution``, require the published observation to be Valid.
        With it, resolve the same inputs without changing their publication.
        Both paths reject revoked resources and notification
        reentry; a non-Valid result raises CfgPreconditionError.
        """
        ...


@dataclass(frozen=True)
class CfgResolution:
    """One command's fixed source view; callbacks must never perform live I/O.

    Composition binds all ports to the same detached source snapshots. Editing
    and refresh acquire a new view. Observe, watch and ordinary acceptance do
    not. A Run owner may supply a one-off view to acceptance without publishing
    it. ``source_basis`` identifies those snapshots; the other fields evaluate
    expressions, enumerate options, resolve references, read frozen values and
    validate expression syntax, respectively.
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
    fresh creation/reset and must return the same definition throughout this
    lifetime. Restore supplies ``initial`` instead of executing defaults. Input
    snapshots and replacement belong only to the concrete application owner.
    """

    def __init__(
        self,
        defaults: Callable[[], CfgSchema],
        *,
        resolution: Callable[[], CfgResolution],
        initial: CfgSchema | None = None,
        mutation_allowed: Callable[[], bool] = lambda: True,
        notification_active: Callable[[], bool] = lambda: False,
    ) -> None:
        self._defaults = defaults
        self._resolution = resolution
        self._mutation_allowed = mutation_allowed
        self._notification_active = notification_active
        self._closed = False
        self._notifying = 0
        self._subscribers: dict[object, Callable[[CfgObservation], None]] = {}
        schema = deepcopy(defaults() if initial is None else initial)
        _validate_definition(schema.spec)
        self._spec = deepcopy(schema.spec)
        self._inputs = deepcopy(schema)
        self._draft, basis = self._prepare(schema)
        self._observation = self._observe_draft(self._draft, basis)

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

    def edit(self, edits: tuple[CfgEdit, ...]) -> CfgObservation:
        from ._input_edit import InputEditor

        edits = _validate_edits(edits)
        self._check_command()
        source = self._resolution()
        editor = InputEditor(self._inputs, source)
        for index, edit in enumerate(edits):
            try:
                editor.apply(edit)
            except CfgInputError as exc:
                raise CfgInputError(
                    exc.reason, str(exc), path=edit.path, edit_index=index
                ) from exc
            except CfgPreconditionError as exc:
                raise CfgPreconditionError(
                    exc.reason, str(exc), path=edit.path, edit_index=index
                ) from exc
        return self._commit_inputs(editor.schema, source)

    def select_custom_reference(self, path: CfgPath, label: str) -> CfgObservation:
        """Select Custom atomically, inheriting the current publication's inputs.

        This is an explicit selection command, not incomplete ordinary edit
        admission. It uses the cached published contents, so detaching a linked
        reference does not require resolving its old source again.
        """
        from ._input_edit import InputEditor
        from .inheritance import detach_input_tree

        _validate_path(path)
        self._check_command()
        source = self._resolution()
        published = self._draft.snapshot()
        published.value = detach_input_tree(published.value)
        editor = InputEditor(published, source)
        try:
            editor.select_custom_reference(path, label)
        except CfgInputError as exc:
            raise CfgInputError(exc.reason, str(exc), path=path) from exc
        return self._commit_inputs(editor.schema, source)

    def _commit_inputs(
        self, inputs: CfgSchema, source: CfgResolution
    ) -> CfgObservation:
        candidate, basis = self._prepare(deepcopy(inputs), source=source)
        try:
            observation = self._observe_draft(candidate, basis)
        except BaseException:
            candidate.close()
            raise
        return self._publish(candidate, observation, inputs)

    def require_mutation_allowed(self) -> None:
        """Guard an owning flow before it commits related non-cfg state."""
        self._check_command()

    def snapshot_inputs(self) -> CfgSchema:
        """Detach input state for the owning persistence/load flow, without refresh."""
        self._require_open()
        return deepcopy(self._inputs)

    def replace_inputs(
        self,
        schema: CfgSchema,
        *,
        require_valid: bool = False,
    ) -> CfgObservation:
        """Install owning-flow inputs; optional backfill can require Valid."""
        self._check_command()
        return self._replace_inputs(deepcopy(schema), require_valid=require_valid)

    def reset(self) -> CfgObservation:
        self._check_command()
        return self._replace_inputs(deepcopy(self._defaults()))

    def _replace_inputs(
        self, schema: CfgSchema, *, require_valid: bool = False
    ) -> CfgObservation:
        if schema.spec != self._spec:
            raise RuntimeError("replacement changed the resource definition")
        candidate, basis = self._prepare(deepcopy(schema))
        try:
            observation = self._observe_draft(candidate, basis)
            if require_valid and observation.status is not CfgStatus.VALID:
                raise CfgInputError(
                    CfgInputReason.INVALID_VALUE,
                    "Owning-flow replacement requires a Valid candidate",
                )
        except BaseException:
            candidate.close()
            raise
        return self._publish(candidate, observation, schema)

    def refresh(self) -> CfgObservation:
        self._check_command(editing=False)
        return self.refresh_group((self,), resolution=self._resolution)[0]

    def accept(
        self,
        *,
        resolution: CfgResolution | None = None,
    ) -> AcceptedConfig:
        """Return detached execution values for the current inputs.

        Without ``resolution``, use only the cached Valid publication. A supplied
        fixed resolution prepares the current inputs as a temporary candidate,
        including when the publication is Invalid or Unavailable. It never
        changes inputs, subscribers or the source factory. Non-Valid
        results raise NOT_VALID; revoked and notification guards apply
        before evaluating the candidate. Unexpected resolver errors propagate.
        """
        self._check_command(editing=False)
        if resolution is not None:
            candidate, basis = self._prepare(deepcopy(self._inputs), source=resolution)
            try:
                if not candidate.is_valid():
                    raise CfgPreconditionError(
                        CfgPreconditionReason.NOT_VALID, "cfg is not valid"
                    )
                values = lower_resolved_cfg(candidate.snapshot())
                return AcceptedConfig(basis, values)
            finally:
                candidate.close()
        if self._observation.status is not CfgStatus.VALID:
            raise CfgPreconditionError(
                CfgPreconditionReason.NOT_VALID, "cfg is not valid"
            )
        values = lower_resolved_cfg(self._draft.snapshot())
        return AcceptedConfig(self._observation.source_basis, deepcopy(values))

    def preview(self, *, resolution: CfgResolution) -> CfgObservation:
        """Resolve current inputs against a fixed source without publication.

        App owners receive a detached Valid or Invalid observation with the
        supplied source basis. Inputs, publication, subscribers
        and the source factory stay unchanged. Revoked and notification
        guards run before evaluation; unexpected resolver failures propagate.
        This does not accept execution values or refresh the publication.
        """
        self._check_command(editing=False)
        candidate, basis = self._prepare(deepcopy(self._inputs), source=resolution)
        try:
            return self._observe_draft(candidate, basis)
        finally:
            candidate.close()

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
        if self._notifying or self._notification_active():
            raise CfgPreconditionError(
                CfgPreconditionReason.REENTRANT_MUTATION,
                "cfg mutation and acceptance are forbidden during notification",
            )

    def _check_command(self, *, editing: bool = True) -> None:
        self._require_open()
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
    ) -> tuple[CfgDraft, SourceBasis]:
        source = self._resolution() if source is None else source
        failures: list[Exception] = []

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

    @staticmethod
    def _observe_draft(draft: CfgDraft, basis: SourceBasis) -> CfgObservation:
        tree = draft.observe()
        diagnostics = tuple(_diagnostics(tree))
        status = CfgStatus.VALID if tree.valid else CfgStatus.INVALID
        return CfgObservation(status, tree, basis, diagnostics)

    def _publish(
        self,
        candidate: CfgDraft,
        observation: CfgObservation,
        inputs: CfgSchema | None = None,
        *,
        notify: bool = True,
    ) -> CfgObservation:
        previous = self._draft
        if inputs is not None:
            self._inputs = inputs
        self._draft = candidate
        self._observation = observation
        previous.close()
        if notify:
            self._notify()
        return deepcopy(observation)

    def _unavailable_observation(
        self, failure: Exception, source: CfgResolution | None
    ) -> CfgObservation:
        from ._unavailable import unavailable_tree

        return CfgObservation(
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

    def _prepare_refresh(
        self, source: CfgResolution
    ) -> tuple[CfgDraft | None, CfgObservation]:
        candidate: CfgDraft | None = None
        try:
            candidate, basis = self._prepare(deepcopy(self._inputs), source=source)
            return candidate, self._observe_draft(candidate, basis)
        except Exception as exc:
            if candidate is not None:
                candidate.close()
            logging.getLogger(__name__).exception("Cfg resolution failed")
            return None, self._unavailable_observation(exc, source)

    def _prepare_own_source_refresh(self) -> tuple[CfgDraft | None, CfgObservation]:
        try:
            source = self._resolution()
        except Exception as exc:
            logging.getLogger(__name__).exception("Cfg source snapshot failed")
            return None, self._unavailable_observation(exc, None)
        return self._prepare_refresh(source)

    def _install_refresh(
        self, candidate: CfgDraft | None, observation: CfgObservation
    ) -> None:
        if candidate is None:
            self._observation = observation
        else:
            self._publish(candidate, observation, notify=False)

    @staticmethod
    def notifications_active(resources: Sequence[CfgResource]) -> bool:
        """Keep notification-state inspection inside the cfg implementation owner."""
        return any(resource._notifying for resource in resources)

    @staticmethod
    def refresh_group(
        resources: Sequence[CfgResource],
        *,
        resolution: Callable[[], CfgResolution] | None = None,
    ) -> tuple[CfgObservation, ...]:
        """Prepare all publications, install all, then notify subscribers.

        A supplied resolution is one shared source. None uses each resource\'s
        own source factory, preserving fixed consumer captures. Source failures
        publish Unavailable per affected resource; notification is never partial.
        """
        members = tuple(resources)
        if not members:
            return ()
        for resource in members:
            resource._check_command(editing=False)
        source: CfgResolution | None = None
        failure: Exception | None = None
        if resolution is not None:
            try:
                source = resolution()
            except Exception as exc:
                failure = exc
                logging.getLogger(__name__).exception("Cfg source snapshot failed")
        prepared: list[tuple[CfgResource, CfgDraft | None, CfgObservation]] = []
        for resource in members:
            candidate: CfgDraft | None = None
            if failure is not None:
                observation = resource._unavailable_observation(failure, None)
            elif resolution is None:
                candidate, observation = resource._prepare_own_source_refresh()
            else:
                assert source is not None
                candidate, observation = resource._prepare_refresh(source)
            prepared.append((resource, candidate, observation))
        for resource, candidate, observation in prepared:
            resource._install_refresh(candidate, observation)
        for resource in members:
            resource._notifying += 1
        try:
            for resource in members:
                resource._notify()
        finally:
            for resource in members:
                resource._notifying -= 1
        return tuple(deepcopy(observation) for _, _, observation in prepared)


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
    value = tree.value
    if isinstance(value, ReferenceValue) and value.error:
        yield CfgDiagnostic(path, "invalid_input", value.error)
    elif not tree.valid and not tree.children:
        message = "input is incomplete or invalid"
        if isinstance(value, (DirectValue, EvalValue)):
            message = value.error or value.validation_error or message
        elif isinstance(value, SweepValue):
            errors = [
                endpoint.error or endpoint.validation_error
                for endpoint in (value.start, value.stop)
                if isinstance(endpoint, (DirectValue, EvalValue))
                and (endpoint.error or endpoint.validation_error)
            ]
            message = (
                "; ".join(error for error in errors if error is not None) or message
            )
        yield CfgDiagnostic(path, "invalid_input", message)
    for key, child in tree.children.items():
        yield from _diagnostics(child, (*path, key))

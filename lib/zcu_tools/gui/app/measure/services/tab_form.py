"""Measure policy for one headless FormState and its fixed source lifetime."""

from __future__ import annotations

import json
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger
from typing import Literal

from pydantic import JsonValue

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.gui.app.measure.services.form_contracts import (
    CfgPreconditionError,
    CfgPreconditionReason,
    SourceBasis,
)
from zcu_tools.gui.form import (
    FormDescription,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormPreview,
    FormState,
    FormWritePath,
)
from zcu_tools.gui.form_adapter import (
    ElementsObservation,
    FieldDiagnostic,
    Form,
    FormContext,
    LogicalEdit,
    LogicalEditCandidate,
    LogicalFieldObservation,
    ModuleValues,
)

from .tab_input import TabInput

_logger = getLogger(__name__)


@dataclass(frozen=True)
class TabFormSources:
    """One owner-captured source used from Open or Reset until the next Reset.

    env borrows fixed parameter/library snapshots and detached cached device infos.
    source_basis describes their provenance; empty means no revision provenance
    was supplied. Callers must not mutate these captured sources.
    """

    env: FormEnv
    source_basis: SourceBasis = ()


@dataclass(frozen=True)
class AcceptedFormConfig:
    """Detached accepted execution values and the fixed sources that produced them.

    cfg is the validated concrete core cfg. values is its ordinary Python field
    tree for the adapter Run interface. source_basis is capture provenance, not
    a revision precondition. None of these containers alias editable input.
    """

    cfg: ConfigBase
    values: dict[str, ExperimentCfgValue]
    source_basis: SourceBasis


class TabFormResource:
    """Own a tab's inputs and captured sources independently of any frontend.

    Open and explicit Reset capture once. Input/describe/preview/result/accept do
    not refresh sources. Only current valid input may be accepted. Frontends
    send commands here; editor_state lends the current FormState for reading
    and subscribing only. Owner-sequence use is required. Mutation guards and
    notification reentrancy apply before editing; revocation permanently
    invalidates retained handles.
    """

    def __init__[C: ConfigBase](
        self,
        make_form: Callable[[FormContext], Form[C]],
        *,
        capture: Callable[[Mapping[str, str] | None], TabFormSources],
        mutation_allowed: Callable[[], bool],
        notification_active: Callable[[], bool] = lambda: False,
        initial: Mapping[str, JsonValue] | TabInput | None = None,
    ) -> None:
        """Prepare default input using one capture before exposing the resource.

        make_form builds a definition from the newly captured context on Open
        and Reset. The caller binds its template catalog before returning it.
        capture accepts one-shot named alias overrides, or None for global rows.
        mutation_allowed is the app busy/control policy; notification_active
        reports notifications in other tab resources. Neither callback may do
        hardware I/O. initial optionally supplies fixed FormInput {data,selections}
        JSON, either as a detached mapping or opaque TabInput from to_json().
        The form owner decodes with FormInput.from_json and recursively restores
        on fresh defaults; dropped_paths reports discarded paths. Malformed JSON
        raises ValueError. Source captures are not restored. Caller retains its
        input; capture/default failures propagate without a resource.
        """
        self._capture = capture
        self._make_form = make_form
        self._mutation_allowed = mutation_allowed
        self._notification_active = notification_active
        self._sources = capture(None)
        self._form = make_form(FormContext(self._sources.env.parameters))
        self._state = FormState(self._form.default_input(self._sources.env))
        self._dropped_paths = (
            self._state.restore_input(
                FormInput.from_json(
                    self._state.input.model,
                    initial.form_json
                    if isinstance(initial, TabInput)
                    else json.dumps(initial),
                ),
                self._sources.env,
            )
            if initial is not None
            else ()
        )
        self._open = True
        self._notifying = False
        self._callbacks: list[Callable[[], None]] = []

    @property
    def input(self) -> FormInput[FormModel]:
        """Return detached current authored input, including unfinished expressions."""
        self._require_open()
        return self._state.input

    @property
    def dropped_paths(self) -> tuple[FormPath, ...]:
        """Return paths discarded while overlaying initial saved compatible input."""
        self._require_open()
        return self._dropped_paths

    @property
    def env(self) -> FormEnv:
        """Borrow the fixed environment read-only; only Reset replaces it."""
        self._require_open()
        return self._sources.env

    @property
    def editor_state(self) -> FormState[FormModel]:
        """Lend the current FormState to a FormEditor for reading and subscribing.

        This is the same object the resource edits, not a copy or mirror, so an
        editor built over it shows every committed write, and its subscription
        fires on each one. A command that replaces the input, such as Reset,
        installs a new FormState: watch this resource through subscribe and
        re-read the property afterwards. All form writes and Reset go through
        the resource's commands. Calling
        state.write or state.run_command on the lent state bypasses busy,
        lifetime and notification guards and subscriber publication, breaking
        this contract.
        Raises CfgPreconditionError(RESOURCE_GONE) after revoke.
        """
        self._require_open()
        return self._state

    def describe(self, path: FormWritePath = ()) -> FormDescription:
        """Describe one current layer using fixed sources, without publication.

        path is canonical model-field names, () for root. Unknown/inactive
        paths and descendants of whole leaves raise FormPathError with the full
        request path. Other declaration failures retain their exception types. Revocation
        raises RESOURCE_GONE; busy and notification reads are allowed. This
        does not capture sources, materialize refs, or install defaults.
        """
        self._require_open()
        return self._state.describe(path, self._sources.env)

    def preview(self) -> FormPreview:
        """Return partial ordinary leaves and validation errors without publication."""
        self._require_open()
        return self._state.preview(self._sources.env)

    def result(self) -> ConfigBase | FormErrors:
        """Validate current inputs and lower to the concrete cfg, or directed errors.

        No previous valid result is retained. Author lower-hook failures propagate;
        model and core cfg validation failures return FormErrors.
        """
        self._require_open()
        result = self._state.result(self._sources.env)
        if isinstance(result, FormErrors):
            return result
        return self._form.to_cfg(result, self._sources.env)

    def diagnostics(self) -> tuple[FieldDiagnostic, ...]:
        """Return all current canonical errors and seed facts across every element.

        Merges preview and lowered cfg errors, without capture, defaults, edits
        or UI selection. Unknown calibration stays None. Revocation raises the
        same RESOURCE_GONE precondition as other reads.
        """
        self._require_open()
        preview = self.preview()
        result = self.result()
        messages = dict(preview.errors.messages) if preview.errors else {}
        if isinstance(result, FormErrors):
            messages.update(result.messages)
        return self._form.diagnostics(
            self.input, self._sources.env, FormErrors(messages) if messages else None
        )

    def accept(self) -> AcceptedFormConfig:
        """Accept only the current valid cfg without a new source capture.

        Raise CfgPreconditionError(NOT_VALID) for invalid input, RESOURCE_GONE
        after revoke, or REENTRANT_MUTATION during any tab notification. Busy
        operation admission remains with the application operation owner.
        """
        self._require_command(editing=False)
        cfg = self.result()
        if isinstance(cfg, FormErrors):
            raise CfgPreconditionError(
                CfgPreconditionReason.NOT_VALID,
                f"Form is not valid: {dict(cfg.messages)}",
            )
        detached = cfg.model_copy(deep=True)
        values: dict[str, ExperimentCfgValue] = detached.model_dump(mode="python")
        return AcceptedFormConfig(
            detached, values, deepcopy(self._sources.source_basis)
        )

    def write(self, batch: Mapping[FormWritePath, JsonValue]) -> None:
        """Atomically write model-field leaves against the fixed environment.

        FormState's structural/source errors propagate without a partial batch.
        Ordinary invalid values remain editable and make result return errors.
        Busy, revoked or notification-reentrant calls raise CfgPreconditionError.
        """
        with self._mutation():
            self._state.write(batch, self._sources.env)
            self._notify()

    def select(self, path: FormWritePath, selection: JsonValue) -> None:
        """Select one listed option under the same busy/lifetime/publication guard.

        Only this operation executes the declaring converter. Value writes
        remain exact raw replacements. Failure installs/publishes nothing.
        """
        with self._mutation():
            self._state.select(path, selection, self._sources.env)
            self._notify()

    def read_recipe_field(self, path: FormWritePath) -> LogicalFieldObservation:
        """Read one logical field using current input and the fixed capture.

        path contains field names without generated wrappers; () reads root.
        Ref descendants project captured Spec values, never writable drafts.
        Busy, observer and notification reads are allowed. Revocation raises
        RESOURCE_GONE. Adapter path/source failures propagate without changing
        input, base, library, notifications or publication. No capture occurs.
        """
        self._require_open()
        return self._form.read_logical_field(self._state.input, path, self._sources.env)

    def edit_recipe_fields(
        self, edits: Sequence[LogicalEdit], *, require_valid: bool = False
    ) -> bool | FormErrors:
        """Admit and install one ordered logical batch under all mutation guards.

        edits retains duplicate paths and request indexes for adapter admission.
        Structural/source/ref-ancestor failures leave the entire resource intact.
        With require_valid=True, validate the detached candidate through Form
        result and native cfg lowering before installation. Return complete logical
        FormErrors without installation/publication when it is invalid. Otherwise
        ordinary invalid drafts are installed, not rolled back. Return whether
        input changed, not whether it is valid. Changed input notifies state and
        resource subscribers once, preserving base and the fixed source capture.
        Busy, revoked and notification-reentrant commands reject even empty edits.
        """
        with self._mutation():
            candidate = self._form.prepare_logical_edits(
                self._state.input, edits, self._sources.env
            )
            if require_valid:
                result = FormState(candidate.input).result(self._sources.env)
                if not isinstance(result, FormErrors):
                    result = self._form.to_cfg(result, self._sources.env)
                if isinstance(result, FormErrors):
                    errors = self._form.logical_errors(
                        candidate.input, result, self._sources.env
                    )
                    if errors is not None:
                        return errors
            return self._install_recipe_candidate(candidate)

    def select_recipe_module(
        self,
        path: FormWritePath,
        action: Literal["use_library", "use_inline", "disable"],
        *,
        key: str | None = None,
    ) -> bool:
        """Select/copy/disable a logical generated selector in one guarded turn.

        path is nonempty. use_library requires a nonblank actual key; other
        actions require key=None. Adapter owns choices, family, declaration
        locks, nullable state, ref ancestors and captured source admission.
        use_inline copies that capture; existing inline is a guarded no-op.
        Missing/disabled/unresolved selections never create defaults. Return
        changed, not validity. Failures preserve input/base/publication; changed
        candidates notify once. The candidate's logical field observation is
        prepared before installation so a projection failure cannot leave a
        successful selection behind an error reply. Subsequent commands are not
        a transaction.
        """
        with self._mutation():
            if action == "use_library":
                if key is None or not key.strip():
                    raise ValueError("use_library requires a nonempty key")
                candidate = self._form.prepare_library_selection(
                    self._state.input, path, key, self._sources.env
                )
            elif key is not None:
                raise ValueError("key is only accepted for use_library")
            elif action == "use_inline":
                candidate = self._form.prepare_inline_copy(
                    self._state.input, path, self._sources.env
                )
            elif action == "disable":
                candidate = self._form.prepare_module_disable(
                    self._state.input, path, self._sources.env
                )
            else:
                raise ValueError(f"Unknown selector action: {action}")
            self._form.read_logical_field(candidate.input, path, self._sources.env)
            return self._install_recipe_candidate(candidate)

    def recipe_errors(self) -> FormErrors | None:
        """Return complete logical Form preview errors, without cfg lowering.

        Uses current input and fixed sources in this owner turn. This does not
        mutate, capture or certify cfg validity. Revocation raises RESOURCE_GONE.
        """
        self._require_open()
        return self._form.logical_errors(
            self._state.input,
            self._state.preview(self._sources.env).errors,
            self._sources.env,
        )

    def _install_recipe_candidate(self, candidate: LogicalEditCandidate) -> bool:
        if candidate.changed:
            self._state.install_input(candidate.input)
            self._notify()
        return candidate.changed

    def read_elements(
        self, path: FormWritePath, *, fields: tuple[FormWritePath, ...] = ()
    ) -> ElementsObservation:
        """Return all original elements in one readonly turn using fixed sources.

        path is a canonical model-list field. The Form owner returns detached
        authored input, partial values and directed errors for every element,
        independent of picker/participants. Busy and notification reads are
        allowed; revoked handles raise RESOURCE_GONE. Does not capture, convert,
        lower native cfg, install input, notify or change UI focus.
        fields requests nonempty relative logical paths, returned per element
        in request order from this same input and capture. Invalid paths raise
        FormPathError; unavailable fields retain directed errors and identity.
        """
        self._require_open()
        return self._form.read_elements(
            self._state.input, path, self._sources.env, fields=fields
        )

    def use_inline(self, path: FormWritePath) -> None:
        """Copy one canonical core module from its capture under mutation guards.

        path contains str field names and int original model-list positions.
        Inline is a guarded no-op, even for invalid drafts. Ref invokes the same
        to_inline command as the editor. Missing/disabled/nonmodule fields raise
        ValueError; busy, revoked and reentrant calls reject before inspecting
        input. Only a successful copy publishes, once. No source refresh occurs.
        """
        with self._mutation():
            if self._form.module_input_kind(self._state.input, path) == "ref":
                self._state.run_command(path, "to_inline", self._sources.env)
                self._notify()

    def read_module_kind(self, path: FormWritePath) -> Literal["ref", "inline"]:
        """Read one authored core module's tag without evaluating its draft.

        path contains str core field names and int original model-list indexes,
        for example ("tones", 1, "drive"), not inline descendants or extras. Return ref or inline,
        even for partial input or invalid expressions. Unknown/nonmodule paths,
        disabled, missing or untagged input raise ValueError naming the field.
        Revocation raises CfgPreconditionError(RESOURCE_GONE). Busy and
        notification reads are allowed; mutation callers must first use
        require_mutation_allowed. No source read, expression evaluation,
        validation, input change or publication occurs.
        """
        self._require_open()
        return self._form.module_input_kind(self.input, path)

    def read_module(self, path: FormWritePath) -> ModuleValues:
        """Read one core module field using the fixed Open/Reset capture.

        path contains str core field names and int original model-list indexes,
        not inline leaf paths. Unknown/inactive paths raise FormPathError;
        a known nonmodule field raises ValueError. Return detached field-local values
        and origin, not native cfg after to_cfg or extras lowering. Other invalid
        fields do not block reads; this module's errors raise
        CfgPreconditionError(NOT_VALID). Revocation raises RESOURCE_GONE.
        Busy/notification reads are allowed. No capture, input change or
        publication occurs. Unexpected source failures propagate.
        """
        self._require_open()
        result = self._form.read_module(self.input, path, self._sources.env)
        if isinstance(result, FormErrors):
            raise CfgPreconditionError(
                CfgPreconditionReason.NOT_VALID,
                f"Module is not valid: {dict(result.messages)}",
            )
        return result

    def run_command(self, path: FormWritePath, name: str) -> None:
        """Execute a current option's named command against the fixed capture.

        path is a typed tuple of field names/model-list indexes identifying an active field.
        name comes from field commands or selected option commands, for example
        run_command(("modules", "qub_pulse"), "to_inline"). The Form owner
        computes the replacement; this host never copies or refreshes sources.
        Success commits one atomic FormState write and publishes once through
        the same resource subscribers and cfg version authority as write.
        Unknown paths, unmatched options and unavailable commands raise
        ValueError. Source/callable/structural failures propagate with no input
        change or publication. Ordinary invalid results remain editable drafts.
        Busy, revoked and notification-reentrant calls raise CfgPreconditionError
        before command resolution, including commands that would do no work.
        editor_state is read-only to callers; editors must use this entrypoint.
        """
        with self._mutation():
            self._state.run_command(path, name, self._sources.env)
            self._notify()

    def restore_input(self, saved: FormInput[FormModel]) -> tuple[FormPath, ...]:
        """Restore saved authored input and report dropped paths using Core rules.

        Preserve expression text, invalid leaf values and accepted explicit
        selections. Busy/lifetime guards match write, but restoration does not
        run owner input-change checks. No source is recaptured.
        """
        with self._mutation():
            dropped = self._state.restore_input(saved, self._sources.env)
            self._notify()
            return dropped

    def refresh_sources(self, *, aliases: Mapping[str, str] | None = None) -> None:
        """Recapture sources and revalidate without changing any authored input.

        Call on the owner thread. aliases optionally overrides named bindings
        for this capture only; it never changes the global table. Expression
        drafts, explicit selections, resolved direct defaults and their original
        seed sources/metadata stay intact, including invalid input. Subscribers receive one publication
        and can preview against the new capture. Capture or form-factory failures
        preserve the previous input, definition and sources. Busy, revoked and
        notification-reentrant calls raise CfgPreconditionError.
        """
        with self._mutation():
            sources = self._capture(aliases)
            form = self._make_form(FormContext(sources.env.parameters))
            current = self._state.input
            form.seed_sources = self._form.seed_sources
            form.seed_metadata = self._form.seed_metadata
            state = FormState(
                FormInput(form.model, current.data, selections=current.selections)
            )
            self._sources = sources
            self._form = form
            self._state = state
            self._notify()

    def reset(self, *, aliases: Mapping[str, str] | None = None) -> None:
        """Capture sources once and replace input with fresh authored defaults.

        aliases optionally overrides named bindings for this capture only. It is
        neither retained as an override nor written to the global alias table.
        Capture/factory/default failures preserve the old shape, input and source.
        A successful Reset replaces shape and defaults together. Busy, revoked
        and notification-reentrant calls raise CfgPreconditionError.
        """
        with self._mutation():
            sources = self._capture(aliases)
            form = self._make_form(FormContext(sources.env.parameters))
            state = FormState(form.default_input(sources.env))
            self._sources = sources
            self._form = form
            self._state = state
            self._notify()

    def require_mutation_allowed(self) -> None:
        """Guard related owner operations such as Load before replacing results."""
        self._require_command(editing=True)

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Watch committed writes/Reset; return idempotent unsubscribe.

        No initial callback is delivered. Observations are allowed during delivery;
        mutation, acceptance and lifetime changes are not. Subscriber exceptions
        are logged with traceback after installation; delivery continues to other
        subscribers without retry or rollback of the committed input.
        """
        self._require_open()
        self._callbacks.append(callback)
        active = True

        def unsubscribe() -> None:
            nonlocal active
            if active:
                active = False
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe

    def revoke(self) -> None:
        """Permanently invalidate retained handles; notification reentrancy rejects."""
        self._require_command(editing=False)
        self._open = False
        self._callbacks.clear()

    @staticmethod
    def notifications_active(resources: Sequence[TabFormResource]) -> bool:
        """Report whether any supplied resource is delivering owner notifications."""
        return any(resource._notifying for resource in resources)

    def _require_open(self) -> None:
        if not self._open:
            raise CfgPreconditionError(
                CfgPreconditionReason.RESOURCE_GONE, "Tab form resource was retired"
            )

    def _require_command(self, *, editing: bool) -> None:
        self._require_open()
        if self._notifying or self._notification_active():
            raise CfgPreconditionError(
                CfgPreconditionReason.REENTRANT_MUTATION,
                "Tab form command is forbidden during notification",
            )
        if editing and not self._mutation_allowed():
            raise CfgPreconditionError(
                CfgPreconditionReason.MUTATION_BLOCKED, "Tab form editing is blocked"
            )

    @contextmanager
    def _mutation(self) -> Generator[None]:
        # One guard spans installation, editor callbacks and resource publication.
        self.require_mutation_allowed()
        self._notifying = True
        try:
            yield
        finally:
            self._notifying = False

    def _notify(self) -> None:
        for callback in tuple(self._callbacks):
            try:
                callback()
            except Exception:
                _logger.exception("Tab form subscriber failed")


class TabFormResources:
    """Manage State-owned form identity, busy policy and retained handle revocation.

    Composition owns this pool. Frontends resolve a tab-bound resource and issue
    its public commands; they do not construct or retire editable state.
    """

    def __init__(self, *, mutation_allowed: Callable[[str], bool]) -> None:
        """Inject the application tab busy/control policy, evaluated per command."""
        self._mutation_allowed = mutation_allowed
        self._resources: dict[str, TabFormResource] = {}

    def create[C: ConfigBase](
        self,
        tab_id: str,
        make_form: Callable[[FormContext], Form[C]],
        *,
        capture: Callable[[Mapping[str, str] | None], TabFormSources],
        initial: Mapping[str, JsonValue] | TabInput | None = None,
    ) -> TabFormResource:
        """Prepare one fixed capture and optional compatible restoration before registration.

        tab_id is a nonempty new app identity. make_form builds a catalog-bound
        definition; capture reads published sources on Open/Reset only. initial
        is native Form input JSON or opaque TabInput, never a cfg/result. Owner
        thread only; caller retains input, pool owns the resource. Duplicate/empty
        identities raise ValueError; preparation failures leave no association.
        Notifications in any pooled resource forbid lifetime mutation.
        """
        self._require_not_notifying()
        if not tab_id or tab_id in self._resources:
            raise ValueError(
                f"Tab form identity is empty or already registered: {tab_id!r}"
            )
        resource = TabFormResource(
            make_form,
            capture=capture,
            initial=initial,
            mutation_allowed=lambda: self._mutation_allowed(tab_id),
            notification_active=self._notifications_active,
        )
        self._resources[tab_id] = resource
        return resource

    def lookup(self, tab_id: str) -> TabFormResource:
        """Resolve the tab's sole form authority; missing identity raises RESOURCE_GONE."""
        try:
            return self._resources[tab_id]
        except KeyError as exc:
            raise CfgPreconditionError(
                CfgPreconditionReason.RESOURCE_GONE,
                f"Tab {tab_id!r} has no form resource",
            ) from exc

    def retire(self, tab_id: str) -> None:
        """Revoke old handles before removing identity; the app checks close/busy policy."""
        self.lookup(tab_id).revoke()
        del self._resources[tab_id]

    def _notifications_active(self) -> bool:
        return TabFormResource.notifications_active(tuple(self._resources.values()))

    def _require_not_notifying(self) -> None:
        if self._notifications_active():
            raise CfgPreconditionError(
                CfgPreconditionReason.REENTRANT_MUTATION,
                "Tab lifetime changes are forbidden during form notification",
            )

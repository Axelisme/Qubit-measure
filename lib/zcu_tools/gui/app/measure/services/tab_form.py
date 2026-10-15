"""Measure policy for one headless FormState and its fixed source lifetime."""

from __future__ import annotations

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
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormPreview,
    FormState,
    FormWritePath,
)
from zcu_tools.gui.form_adapter import Form, FormContext, ModuleValues

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

    Open and explicit Reset capture once. Input/schema/preview/result/accept do
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
        initial: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Prepare default input using one capture before exposing the resource.

        make_form builds a definition from the newly captured context on Open
        and Reset. The caller binds its template catalog before returning it.
        capture accepts one-shot named alias overrides, or None for global rows.
        mutation_allowed is the app busy/control policy; notification_active
        reports notifications in other tab resources. Neither callback may do
        hardware I/O. initial optionally overlays saved compatible input on fresh
        defaults; dropped_paths reports discarded leaves. Source captures are not
        restored. Capture/default failures propagate without a resource.
        """
        self._capture = capture
        self._make_form = make_form
        self._mutation_allowed = mutation_allowed
        self._notification_active = notification_active
        self._sources = capture(None)
        self._form = make_form(FormContext(self._sources.env.parameters))
        self._state = FormState(self._form.default_input(self._sources.env))
        self._dropped_paths = (
            self._state.write_compatible(initial, self._sources.env)
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

    def input_schema(self) -> dict[str, JsonValue]:
        """Return the generated editable schema using this tab's fixed sources."""
        self._require_open()
        return self._state.input_schema(self._sources.env)

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

    def read_module_kind(self, path: tuple[str, ...]) -> Literal["ref", "inline"]:
        """Read one authored core module's tag without evaluating its draft.

        path contains nonempty core field names, for example ("modules",
        "qub_pulse"), not inline descendants or extras. Return ref or inline,
        even for partial input or invalid expressions. Unknown/nonmodule paths,
        disabled, missing or untagged input raise ValueError naming the field.
        Revocation raises CfgPreconditionError(RESOURCE_GONE). Busy and
        notification reads are allowed; mutation callers must first use
        require_mutation_allowed. No source read, expression evaluation,
        validation, input change or publication occurs.
        """
        self._require_open()
        return self._form.module_input_kind(self.input, path)

    def read_module(self, path: tuple[str, ...]) -> ModuleValues:
        """Read one core module field using the fixed Open/Reset capture.

        path contains nonempty core field names, not inline leaf paths. Unknown
        or nonmodule paths raise ValueError. Return detached field-local values
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

        path is a tuple of model-field names identifying an option field.
        name comes from that selected option's schema commands, for example
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

    def write_compatible(self, saved: Mapping[str, JsonValue]) -> tuple[FormPath, ...]:
        """Overlay compatible saved leaves and report discarded input paths.

        Preserve expression text and unfinished input. FormState owns branch/type
        and source compatibility. Mutation guards match write; failures preserve
        the entire input. No source is recaptured.
        """
        with self._mutation():
            dropped = self._state.write_compatible(saved, self._sources.env)
            self._notify()
            return dropped

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
        initial: Mapping[str, JsonValue] | None = None,
    ) -> TabFormResource:
        """Prepare one fixed capture and optional compatible restoration before registration.

        tab_id is a nonempty new app identity. make_form builds a catalog-bound
        definition; capture reads published sources on Open/Reset only. initial
        is saved Form input JSON, never an old CfgSchema memento. Duplicate/empty
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

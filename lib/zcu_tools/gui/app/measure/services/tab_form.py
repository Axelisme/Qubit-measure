"""Measure policy for one headless FormState and its fixed source lifetime."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from logging import getLogger

from pydantic import JsonValue

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.gui.app.measure.adapter import Form
from zcu_tools.gui.cfg import ExpressionCapture
from zcu_tools.gui.cfg.resource import (
    CfgPreconditionError,
    CfgPreconditionReason,
    SourceBasis,
)
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPreview,
    FormState,
    FormWritePath,
)
from zcu_tools.gui.session.expression import validate_scalar_expr
from zcu_tools.gui.session.parameters import ModuleSlotError
from zcu_tools.resources.entry import RoleResolutionError

_logger = getLogger(__name__)


@dataclass(frozen=True)
class TabFormSources:
    """One owner-captured source used from Open or Reset until the next Reset.

    env borrows fixed parameter/library snapshots and detached native devices.
    captures contains published dotted value-source cache entries, never live
    providers. source_basis describes their provenance; empty means no revision
    provenance was supplied. Callers must not mutate these captured sources.
    """

    env: FormEnv
    captures: Mapping[str, object] = field(default_factory=dict)
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
    not refresh sources. Only current valid input may be accepted. This owner
    does not expose its mutable FormState; frontends send commands here.
    Owner-sequence use is required. Mutation guards and notification reentrancy
    apply before editing; revocation permanently invalidates retained handles.
    """

    def __init__[C: ConfigBase](
        self,
        form: Form[C],
        *,
        capture: Callable[[Mapping[str, str] | None], TabFormSources],
        mutation_allowed: Callable[[], bool],
        notification_active: Callable[[], bool] = lambda: False,
    ) -> None:
        """Prepare default input using one capture before exposing the resource.

        form is the registry-bound generated definition for this tab lifetime.
        capture accepts one-shot named alias overrides, or None for global rows.
        mutation_allowed is the app busy/control policy; notification_active
        reports notifications in other tab resources. Neither callback may do
        hardware I/O. Capture/default failures propagate without a resource.
        """
        self._capture = capture
        self._default_input = form.default_input
        self._to_cfg: Callable[[FormModel, FormEnv], ConfigBase | FormErrors] = (
            form.to_cfg
        )
        self._mutation_allowed = mutation_allowed
        self._notification_active = notification_active
        self._sources = capture(None)
        self._state = FormState(form.default_input(self._sources.env))
        self._open = True
        self._notifying = False
        self._callbacks: list[Callable[[], None]] = []

    @property
    def input(self) -> FormInput[FormModel]:
        """Return detached current authored input, including unfinished expressions."""
        self._require_open()
        return self._state.input

    @property
    def env(self) -> FormEnv:
        """Borrow the fixed environment read-only; only Reset replaces it."""
        self._require_open()
        return self._sources.env

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
        return self._to_cfg(result, self._sources.env)

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
        Expression @{key} tokens are fixed numeric literals before any input is
        installed. Token grammar/value failures raise CfgInputError; missing
        captured keys raise CfgPreconditionError(CAPTURE_UNAVAILABLE), without
        changing input. Busy, revoked or notification-reentrant calls raise
        CfgPreconditionError.
        """
        self.require_mutation_allowed()
        capture = ExpressionCapture(self._read_capture, validate_scalar_expr)
        prepared = {
            path: _capture_input(value, capture) for path, value in batch.items()
        }
        self._state.write(prepared, self._sources.env)
        self._notify()

    def reset(self, *, aliases: Mapping[str, str] | None = None) -> None:
        """Capture sources once and replace input with fresh authored defaults.

        aliases optionally overrides named bindings for this capture only. It is
        neither retained as an override nor written to the global alias table.
        Capture/default failures preserve the old input and source. Busy, revoked
        and notification-reentrant calls raise CfgPreconditionError.
        """
        self.require_mutation_allowed()
        sources = self._capture(aliases)
        state = FormState(self._default_input(sources.env))
        self._sources = sources
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

    def _read_capture(self, key: str) -> object:
        if key in self._sources.captures:
            return self._sources.captures[key]
        parameters = self._sources.env.parameters
        if parameters is not None:
            try:
                item = parameters.lookup(key)
            except (RoleResolutionError, ModuleSlotError) as exc:
                raise KeyError(key) from exc
            if item.present:
                return item.value
        raise KeyError(key)

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

    def _notify(self) -> None:
        self._notifying = True
        try:
            for callback in tuple(self._callbacks):
                try:
                    callback()
                except Exception:
                    _logger.exception("Tab form subscriber failed")
        finally:
            self._notifying = False


def _capture_input(value: JsonValue, capture: ExpressionCapture) -> JsonValue:
    if isinstance(value, dict):
        if set(value) == {"expr"} and isinstance(expression := value["expr"], str):
            return {"expr": capture.prepare(expression)}
        return {key: _capture_input(item, capture) for key, item in value.items()}
    if isinstance(value, list):
        return [_capture_input(item, capture) for item in value]
    return value

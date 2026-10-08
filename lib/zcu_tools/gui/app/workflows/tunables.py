"""Input submission shared by GUI and remote callers, without GUI version guards."""

from collections.abc import Mapping
from math import isfinite

from pydantic import JsonValue

from zcu_tools.experiment.workflows import (
    Actor,
    Engine,
    TunableChange,
    TunablesSnapshot,
)
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    FormValue,
    FormWritePath,
)


class TunablesHost[F: FormModel, C]:
    """Submit one accepted run's complete current input as an atomic batch.

    Call on the GUI owner thread. state initially contains the complete form
    input used to start run_id on engine; field paths match native tunables.
    env is that input's fixed evaluation capture. This host borrows them and
    owns accepted input/base for Apply/Discard, not a GUI revision guard.
    It does not Start, schedule workers, capture sources, materialize native
    cfg, or implement/check control identity. Holder enforcement will be added
    by the shared control-identity owner after that mechanism lands.
    """

    state: FormState[F]

    def __init__(
        self, state: FormState[F], env: FormEnv, engine: Engine[C], run_id: str
    ) -> None:
        """Bind accepted input to an existing run without writing native values.

        Wrong run IDs propagate. Caller supplies input corresponding to the
        native run. Set this draft's base to that input; do not change Engine.
        """
        engine.tunables(run_id)
        self.state = state
        self._env = env
        self._engine = engine
        self._run_id = run_id
        self._accepted_input = state.input
        state.reset(self._accepted_input)

    @property
    def accepted_input(self) -> FormInput[F]:
        """Return latest successfully submitted input, including expressions."""
        return self._accepted_input

    def apply(
        self,
        *,
        actor: Actor,
        local_errors: FormErrors | None = None,
    ) -> TunablesSnapshot | FormErrors:
        """Submit complete current ordinary values, not a draft diff.

        local_errors block submission even when state retains valid values.
        Preview/model errors return FormErrors. Otherwise convert complete
        preview leaves (including defaults) to JSON and submit using Engine's
        current revision, read immediately before its atomic update. GUI/RPC
        callers supply no expected revision. Tuples become JSON lists; complex
        fails rather than stringifying. Native run/lifecycle/validation/I/O or
        concurrent Engine revision failures propagate without retry.
        Success resets base to current input; failure retains draft/base.
        Even unchanged complete input is an accepted new Engine revision.
        """
        if local_errors is not None and local_errors.messages:
            return local_errors
        preview = self.state.preview(self._env)
        if preview.errors is not None:
            return preview.errors
        changes = tuple(
            TunableChange(".".join(path), self._json_value(value))
            for path, value in preview.values.items()
        )
        revision = self._engine.tunables(self._run_id).revision
        snapshot = self._engine.update_tunables(
            self._run_id, changes, expected_revision=revision, actor=actor
        )
        self._accepted_input = self.state.input
        self.state.reset(self._accepted_input)
        return snapshot

    def apply_external(
        self,
        batch: Mapping[FormWritePath, JsonValue],
        *,
        actor: Actor,
    ) -> TunablesSnapshot | FormErrors:
        """Write batch to the current shared draft and submit complete input.

        Existing parsed GUI draft values participate in the submitted batch.
        Structural/source write errors propagate. Model-invalid input remains
        in the draft and returns FormErrors; accepted input/base are unchanged.
        Native failures propagate without retry. Success updates accepted input
        and resets draft base. There is no external-edit draft preservation or
        conflict guard. Caller clears renderer-local text only after success.
        """
        self.state.write(batch, self._env)
        return self.apply(actor=actor)

    def discard(self) -> None:
        """Restore latest accepted expression input without changing Engine.

        Caller separately discards renderer-local unparseable text. No Engine
        revision or holder state is restored or checked.
        """
        self.state.reset(self._accepted_input)

    def _json_value(self, value: FormValue) -> JsonValue:
        if isinstance(value, (list, tuple)):
            return [self._json_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._json_value(item) for key, item in value.items()}
        if isinstance(value, complex):
            raise TypeError("Complex values cannot be submitted as JSON tunables")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("Nonfinite values cannot be submitted as JSON tunables")
        return value

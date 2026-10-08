"""Form input/revision submission policy shared by GUI and remote callers."""

from collections.abc import Mapping

from pydantic import JsonValue

from zcu_tools.experiment.workflows import Actor, Engine, TunablesSnapshot
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    FormWritePath,
)


class TunablesHost[F: FormModel, C]:
    """Manage one accepted run's expression input and a separate GUI draft.

    Call on the GUI owner thread. state must initially contain the complete
    form input used to start run_id on engine, with field paths matching the
    native tunables declaration. env is the fixed evaluation capture used by
    that input. This host borrows state/engine/env; it never starts a run,
    schedules workers, captures sources, or performs native materialization.
    All edits for this run should use this host. Other Engine writers can make
    a draft stale; Apply never repairs or retries that conflict.

    state is the GUI draft. revision identifies its accepted base, not
    necessarily the latest Engine revision. accepted_input retains the latest
    input successfully submitted through this host, including expressions.
    """

    state: FormState[F]

    def __init__(
        self, state: FormState[F], env: FormEnv, engine: Engine[C], run_id: str
    ) -> None:
        """Bind an accepted input to an existing run without writing Engine.

        Engine.run_id errors propagate. The caller supplies input corresponding
        to the current native revision. Construction resets this draft's base
        to that input; it does not overwrite native values.
        """
        self.state = state
        self._env = env
        self._engine = engine
        self._run_id = run_id
        self._revision = engine.tunables(run_id).revision
        self._accepted_revision = self._revision
        self._accepted_input = state.input
        state.reset(self._accepted_input)

    @property
    def revision(self) -> int:
        """Return the revision represented by the GUI draft's base."""
        return self._revision

    @property
    def accepted_input(self) -> FormInput[F]:
        """Return latest successfully submitted input for input persistence."""
        return self._accepted_input

    @property
    def has_conflict(self) -> bool:
        """Return whether an edited GUI draft's base has a stale revision.

        This reads native revision only; it never resets or re-evaluates input.
        """
        raise NotImplementedError

    def apply(
        self,
        *,
        expected_revision: int,
        actor: Actor,
        local_errors: FormErrors | None = None,
    ) -> TunablesSnapshot | FormErrors:
        """Validate and atomically submit the GUI draft's changed leaves.

        local_errors block submission even when retained state is valid.
        Invalid draft returns FormErrors. Otherwise Engine receives ordinary
        values and the exact supplied expected_revision/actor. Empty changes,
        run/lifecycle errors, stale RevisionConflict (expected/actual), native
        validation and I/O errors propagate. Form fields must materialize
        JSON-compatible Engine leaves; tuples become JSON lists, complex values
        fail rather than being stringified. No error changes input/base or
        retries. Success resets base to this input at the returned revision.
        """
        raise NotImplementedError

    def apply_external(
        self,
        batch: Mapping[FormWritePath, JsonValue],
        *,
        expected_revision: int,
        actor: Actor,
        local_errors: FormErrors | None = None,
    ) -> TunablesSnapshot | FormErrors:
        """Submit an input batch independently of pending GUI text/draft.

        Apply batch to the latest accepted expression input via FormState.
        Invalid input returns FormErrors without changing accepted/GUI input.
        Structural/source and Engine failures propagate without retry. Send
        the supplied expected_revision unchanged. Successful input/revision
        becomes accepted_input. A clean GUI draft follows it. An edited GUI
        draft or nonempty local_errors retains its old input/base/revision
        and reports has_conflict; local errors do not block external edits.
        """
        raise NotImplementedError

    def discard(self) -> None:
        """Restore the latest host-accepted input/revision, without Engine edits.

        The caller separately discards renderer-local unparseable text.
        External native writes not made through this host are not reconstructed
        into expression input; a later Apply still detects their stale version.
        """
        raise NotImplementedError

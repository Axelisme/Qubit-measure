"""Input submission shared by GUI and remote callers, without GUI version guards."""

from collections.abc import Callable, Mapping

from pydantic import BaseModel, JsonValue

from zcu_tools.experiment.workflows import (
    Actor,
    Engine,
    TunablesSnapshot,
)
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    FormWritePath,
)


class TunablesHost[F: FormModel, C]:
    """Own one run's fixed form binding before and after native acceptance.

    state is the unique draft for this run; to_cfg maps its validated form to
    the complete native tunables model using the same declaration as Start.
    env is its fixed evaluation capture. The host retains submitted expression
    input/base for Apply/Discard, not a GUI revision guard. Construct it before
    admission, then accept the Engine after native Start succeeds. Apply rejects
    pending acceptance before draft writes. The Controller checks current
    binding identity and owns lifecycle/error cleanup for GUI and RPC callers.

    Use on the owner thread, except accept may run in the startup producer
    under the Controller's binding lock. This host does not Start, schedule,
    capture sources or enforce holder identity.
    """

    state: FormState[F]

    def __init__(
        self,
        state: FormState[F],
        env: FormEnv,
        run_id: str,
        *,
        to_cfg: Callable[[F], BaseModel],
    ) -> None:
        """Retain Start's prepared input and evaluation capture without an Engine.

        state holds complete submitted form input for run_id, the native Start
        identity. env is the corresponding fixed capture. to_cfg converts a
        validated F into the complete native model, without live reads or
        resources; conversion failures propagate without accepting input.
        No native call or source refresh occurs. The initial submitted input becomes the draft's
        base; acceptance remains pending until accept succeeds.
        """
        self.state = state
        self._to_cfg = to_cfg
        self._env = env
        self._engine: Engine[C] | None = None
        self._run_id = run_id
        self._accepted_input = state.input
        state.reset(self._accepted_input)

    @property
    def env(self) -> FormEnv:
        """Return this binding's fixed Start capture, not a live refresh."""
        return self._env

    @property
    def run_id(self) -> str:
        """Return the native identity submitted at construction, even if pending."""
        return self._run_id

    @property
    def engine(self) -> Engine[C] | None:
        """Return the accepted Engine, or None before native acceptance."""
        return self._engine

    def accept(self, engine: Engine[C]) -> None:
        """Attach native acceptance once without resetting draft or writing values.

        engine must contain this run_id with tunables corresponding to Start's
        submitted input. Native target/read failures propagate and leave the
        binding pending. A second attachment raises RuntimeError before writes.
        The Controller calls this under its lock after successful worker Start.
        """
        if self._engine is not None:
            raise RuntimeError("Tunables binding already has native acceptance")
        engine.tunables(self._run_id)
        self._engine = engine

    def _accepted_engine(self) -> Engine[C]:
        """Require native acceptance before preview, draft writes or publication."""
        if self._engine is None:
            raise RuntimeError("Applying tunables requires native acceptance")
        return self._engine

    @property
    def accepted_input(self) -> FormInput[F]:
        """Return latest submitted expression input; pending is not acceptance."""
        return self._accepted_input

    def apply(
        self,
        *,
        actor: Actor,
        local_errors: FormErrors | None = None,
    ) -> TunablesSnapshot | FormErrors:
        """Convert complete current input to the native model and submit it.

        local_errors block submission even when state retains valid values.
        Form/model errors return FormErrors. Otherwise to_cfg uses the same
        declaration as Start, retaining nested models and lists. Read Engine's
        current revision immediately before its atomic update; GUI/RPC supply
        no revision. Native conversion/run/lifecycle/validation/I/O or concurrent
        revision failures propagate without retry, preserving draft/base.
        Only successful publication accepts input and resets its base.
        Pending acceptance raises RuntimeError before evaluation or mutation.
        Even unchanged complete input creates a new Engine revision.
        """
        engine = self._accepted_engine()
        if local_errors is not None and local_errors.messages:
            return local_errors
        result = self.state.result(self._env)
        if isinstance(result, FormErrors):
            return result
        tunables = self._to_cfg(result)
        revision = engine.tunables(self._run_id).revision
        snapshot = engine.update_tunables(
            self._run_id, tunables, expected_revision=revision, actor=actor
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
        Pending acceptance raises RuntimeError before any draft write.
        """
        self._accepted_engine()
        self.state.write(batch, self._env)
        return self.apply(actor=actor)

    def discard(self) -> None:
        """Restore latest accepted expression input without changing Engine.

        Caller separately discards renderer-local unparseable text. No Engine
        revision or holder state is restored or checked.
        """
        self.state.reset(self._accepted_input)

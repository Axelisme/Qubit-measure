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
    """Own one run's fixed form binding before and after native acceptance.

    state is the unique draft for this run; field paths match native tunables.
    env is its fixed evaluation capture. The host retains submitted expression
    input/base for Apply/Discard, not a GUI revision guard. Construct it before
    admission, then accept the Engine after native Start succeeds. Apply rejects
    pending acceptance before draft writes. The Controller checks current
    binding identity and owns lifecycle/error cleanup for GUI and RPC callers.

    Use on the owner thread, except accept may run in the startup producer
    under the Controller's binding lock. This host does not Start, schedule,
    capture sources, materialize cfg or enforce holder identity.
    """

    state: FormState[F]

    def __init__(self, state: FormState[F], env: FormEnv, run_id: str) -> None:
        """Retain Start's prepared input and evaluation capture without an Engine.

        state holds complete submitted form input for run_id, the native Start
        identity. env is the corresponding fixed capture. No native call or
        source refresh occurs. The initial submitted input becomes the draft's
        base; acceptance remains pending until accept succeeds.
        """
        self.state = state
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
        """Submit complete current ordinary values, not a draft diff.

        local_errors block submission even when state retains valid values.
        Preview/model errors return FormErrors. Indexed leaves also return
        FormErrors at their original paths: native tunables accept field-name
        paths only, not list elements. Neither refusal changes Engine, draft,
        base or accepted input. No index stringification or whole-list
        conversion occurs. Otherwise convert complete
        preview leaves (including defaults) to JSON and submit using Engine's
        current revision, read immediately before its atomic update. GUI/RPC
        callers supply no expected revision. Tuples become JSON lists; complex
        fails rather than stringifying. Native run/lifecycle/validation/I/O or
        concurrent Engine revision failures propagate without retry.
        Success resets base to current input; failure retains draft/base.
        Pending acceptance raises RuntimeError before preview or mutation.
        Even unchanged complete input is an accepted new Engine revision.
        """
        engine = self._accepted_engine()
        if local_errors is not None and local_errors.messages:
            return local_errors
        preview = self.state.preview(self._env)
        if preview.errors is not None:
            return preview.errors
        native_values: dict[str, FormValue] = {}
        unsupported: dict[FormWritePath, str] = {}
        for path, value in preview.values.items():
            names: list[str] = []
            for part in path:
                if not isinstance(part, str):
                    unsupported[path] = "Native tunables do not support list elements"
                    break
                names.append(part)
            else:
                native_values[".".join(names)] = value
        if unsupported:
            return FormErrors(unsupported)
        changes = tuple(
            TunableChange(path, self._json_value(value))
            for path, value in native_values.items()
        )
        revision = engine.tunables(self._run_id).revision
        snapshot = engine.update_tunables(
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

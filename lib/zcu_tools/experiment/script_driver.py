"""Non-blocking, single-owner driving for workflow and recipe generators."""

from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

from zcu_tools.core.script_protocol import (
    Answer,
    Ask,
    Operation,
    Question,
    ScriptGenerator,
    ScriptRequest,
    Wait,
)

type ScriptStatus = Literal[
    "running", "waiting", "paused", "completed", "cancelled", "failed"
]
"""Running is ready to drive; waiting needs operation delivery; paused needs an answer.

Completed, cancelled and failed are terminal for driving. Teardown may still
close a failed producer suspended by a host callback error.
"""


def _answer_type(question: Question[Answer]) -> type[Answer]:
    # Authored subclasses have empty args; their parameterized base retains A.
    for base in type(question).__mro__:
        if not issubclass(base, Question):
            continue
        metadata = base.__pydantic_generic_metadata__
        if metadata["origin"] is not Question:
            continue
        args = metadata["args"]
        if args and isinstance(args[0], type) and issubclass(args[0], Answer):
            return args[0]
    raise TypeError("Question must declare Question[AnswerSubclass] before answering")


class ScriptDriver[R]:
    """Own one ScriptGenerator and drive it through request-owned delivery.

    Construction does not execute the generator. Call advance() to run until a
    pending Wait, unanswered Ask or terminal state. A host starts operations and
    updates op.delivery, then calls advance(); the driver never blocks or starts
    work. Every generator resume sends None, never an operation result or answer.

    Use one owner thread and serialize calls. cancel() only records intent;
    it never drives author code. Other driving calls must not reenter. Controls
    become effective at request boundaries, without exception injection.

    Cancel resumes the author at a boundary. The host first cancels and drains
    an active Wait, preserving its real delivery, including partial or a completed
    race. Ask resumes with delivery=None. After cancel, every new Wait goes once
    to deliver_cancelled instead of starting work; every new Ask resumes None.
    That callback must supply a cancelled delivery without re-entering the driver.
    A missing or non-cancelled delivery raises RuntimeError. Return after cancel
    means cancelled and retains the authored result. An operation's cancelled
    delivery alone does not cancel the script. Other exceptions mean failed,
    remain available as error and propagate. close() is for host teardown only;
    it runs finally blocks without promising responses to further yields.
    """

    def __init__(
        self,
        generator: ScriptGenerator[R],
        *,
        deliver_cancelled: Callable[[Operation[object]], None],
    ) -> None:
        """Take ownership of the workflow generator.

        Operation handles and their resources remain owned by the host.
        deliver_cancelled fills an unstarted operation's cancelled delivery on
        this worker. It must not start work, block or re-enter this driver.
        """
        self._generator = generator
        self._deliver_cancelled = deliver_cancelled
        self._status: ScriptStatus = "running"
        self._pending: ScriptRequest | None = None
        self._result: R | None = None
        self._error: Exception | None = None
        self._driving = False
        self._cancelled = False
        self._cancel_requested = False
        self._new_request = False

    @property
    def status(self) -> ScriptStatus:
        """Return running, waiting, paused, completed, cancelled or failed."""
        return self._status

    @property
    def pending_request(self) -> ScriptRequest | None:
        """Return the suspended Wait/Ask, or None before yielding or at terminal."""
        return self._pending

    @property
    def pending_question(self) -> Question[Answer] | None:
        """Return the authored unanswered question while paused, otherwise None."""
        return self._pending.question if isinstance(self._pending, Ask) else None

    @property
    def result(self) -> R | None:
        """Return the original generator return object after return, else None.

        Completed and cooperatively cancelled scripts both retain their return.
        Teardown close and a None return both have result=None.
        """
        return self._result

    @property
    def error(self) -> Exception | None:
        """Return the original driving/cleanup exception on failure, else None."""
        return self._error

    def cancel(self) -> None:
        """Request cooperative cancellation at the next request boundary.

        This never advances author code. The existing workflow owner must drain
        an active operation before advance. Terminal scripts reject new controls.
        """
        if self._status in ("completed", "cancelled", "failed"):
            raise RuntimeError("Cannot cancel a terminal script")
        self._cancel_requested = True

    def pending_answer_type(self) -> type[Answer] | None:
        """Read the pending Ask's declared model without delivery or driving.

        Return None outside an Ask. An unparameterized Question raises TypeError
        just as answering it would. Hosts may project this model's JSON schema.
        """
        question = self.pending_question
        return _answer_type(question) if question is not None else None

    def parse_answer(self, mapping: Mapping[str, object]) -> Answer:
        """Validate wire fields against the pending question's declared Answer.

        This does not deliver or resume. Invalid fields raise ValidationError
        titled with the Question type; no pending Ask raises RuntimeError.
        The same generic inference serves answer() and validate_answer().
        """
        question = self.pending_question
        if question is None:
            raise RuntimeError("No pending question to answer")
        expected = _answer_type(question)
        try:
            return expected.model_validate(mapping)
        except ValidationError as error:
            details: list[InitErrorDetails] = [
                {
                    "type": PydanticCustomError(
                        "answer_validation",
                        "{message}",
                        {"message": item["msg"], "original_type": item["type"]},
                    ),
                    "loc": item["loc"],
                    "input": item["input"],
                }
                for item in error.errors(include_url=False)
            ]
            raise ValidationError.from_exception_data(
                type(question).__name__, details
            ) from error

    def validate_answer(self, answer: Answer) -> None:
        """Check an Answer instance without delivering or running script code.

        No pending Ask raises RuntimeError; the wrong model raises TypeError
        naming the Question and required Answer. All state stays unchanged.
        Hosts may validate before committing their own resume bookkeeping.
        """
        question = self.pending_question
        if question is None:
            raise RuntimeError("No pending question to answer")
        expected = _answer_type(question)
        if not isinstance(answer, expected):
            raise TypeError(
                f"{type(question).__name__}: Answer must be an instance of {expected.__name__}"
            )

    def answer(self, answer: Answer) -> None:
        """Validate an Answer instance, deliver it to the pending Ask and drive.

        The declared Question[A] determines the required model, including its
        subclasses. A wrong model or unparameterized question raises TypeError
        without delivering or resuming; the driver remains paused. No pending
        question raises RuntimeError. After valid delivery, advance() semantics
        apply, including queued cancellation before the next resume.
        """
        if self._driving:
            raise RuntimeError("Cannot answer while driving a script")
        request = self._pending
        if not isinstance(request, Ask):
            raise RuntimeError("No pending question to answer")
        self.validate_answer(answer)
        request.delivery = answer
        self.advance()

    def advance(self) -> None:
        """Run until undelivered Wait/Ask or terminal; repeat pending polls safely.

        Apply cancellation only after a yield. Ready requests resume with None.
        Terminal calls do nothing. Re-entry raises RuntimeError. Unsupported
        yielded requests raise TypeError and fail the script. Script and callback
        exceptions are recorded as failed and re-raised without wrapping.
        """
        if self._driving:
            raise RuntimeError("Cannot re-enter script driving")
        if self._status in ("completed", "cancelled", "failed"):
            return
        self._driving = True
        try:
            self._drive()
        except GeneratorExit:
            self._finish("cancelled")
        except Exception as error:
            self._error = error
            self._finish("failed")
            raise
        finally:
            self._driving = False

    def close(self) -> None:
        """Abandon the generator on its worker during host teardown.

        This is not cooperative cancel: finally runs, but yielded requests receive
        no delivery. A generator that yields while closing raises RuntimeError.
        Cleanup failures are recorded as failed and re-raised. A failed driver
        stays failed after teardown. Completed or cancelled calls do nothing;
        re-entry raises RuntimeError. The host must
        cancel and drain any active operation before closing.
        """
        if self._driving:
            raise RuntimeError("Cannot close while driving a script")
        if self._status in ("completed", "cancelled"):
            return
        self._driving = True
        try:
            self._generator.close()
            self._finish("failed" if self._status == "failed" else "cancelled")
        except Exception as error:
            self._error = error
            self._finish("failed")
            raise
        finally:
            self._driving = False

    def _finish(self, status: ScriptStatus) -> None:
        self._status = status
        self._pending = None

    def _drive(self) -> None:
        while True:
            if self._pending is not None:
                if self._cancel_requested:
                    self._cancelled = True
                request = self._pending
                if self._cancelled:
                    if isinstance(request, Ask):
                        request.delivery = None
                    elif self._new_request and isinstance(request, Wait):
                        self._deliver_cancelled(request.op)
                        delivery = request.op.delivery
                        if delivery is None or not delivery.partial_result:
                            raise RuntimeError(
                                "Owner must supply partial operation delivery"
                            )
                self._new_request = False
                if not self._ready():
                    return
            self._status = "running"
            try:
                request = next(self._generator)
            except StopIteration as done:
                self._result = done.value
                self._finish("cancelled" if self._cancelled else "completed")
                return
            if not isinstance(request, (Wait, Ask)):
                self._generator.close()
                raise TypeError("Script must yield Wait or Ask requests")
            self._pending = request
            self._new_request = True

    def _ready(self) -> bool:
        request = self._pending
        if isinstance(request, Wait) and request.op.delivery is None:
            self._status = "waiting"
            return False
        if (
            isinstance(request, Ask)
            and request.delivery is None
            and not self._cancelled
        ):
            self._status = "paused"
            return False
        return True

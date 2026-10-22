"""Stateless script outcomes and typed question/answer contracts.

These declarations perform no driving, I/O or source lookup. The async host
owns control scheduling and Future delivery. Existing workflow generators still
use the original Wait/Ask contracts until their async migration lands.
"""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, TypeAdapter

type OperationStatus = Literal["completed", "cancelled", "failed"]
"""Terminal outcome: success, cooperative cancellation, or an operation failure."""


class Answer(BaseModel):
    """Base for a host-validated answer to a script's question.

    Subclasses declare all answer fields and their meanings. Unknown fields
    fail validation. Answering does not itself authorize or perform a write.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class Question[A: Answer](BaseModel):
    """Base for an authored question whose reply is an A model.

    Parameterize a subclass with its Answer subclass, then declare the evidence
    and choices as named fields. Hosts expose only explicitly asked questions.
    They validate replies against A before resuming the suspended helper.
    There is no default answer or timeout. Unknown fields fail validation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def parse_answer(self, value: object) -> A:
        """Decode a reply with the authored Answer model, without delivery or I/O."""
        adapter = TypeAdapter[A](self.answer_type())
        return adapter.validate_python(value)

    def validate_answer(self, answer: Answer) -> None:
        """Validate evidence-dependent choices before delivery; default accepts A.

        Subclasses may reject unknown or modified choices with ValueError.
        Validation must not write external state or perform the proposed action.
        """

    def answer_type(self) -> type[Answer]:
        """Return the declared Answer model, including inherited Question bases.

        An unparameterized Question raises TypeError rather than accepting
        arbitrary reply fields. This reflection boundary retains the authored
        generic pairing for both wire validation and typed async delivery.
        """
        for base in type(self).__mro__:
            if not issubclass(base, Question):
                continue
            metadata = base.__pydantic_generic_metadata__
            args = metadata["args"]
            if metadata["origin"] is Question and args:
                candidate = args[0]
                if isinstance(candidate, type) and issubclass(candidate, Answer):
                    return candidate
        raise TypeError(
            "Question must declare Question[AnswerSubclass] before answering"
        )


@dataclass(frozen=True)
class OperationResult[T]:
    """One operation's detached terminal delivery.

    status is completed or failed. result is the operation's named payload, or
    None without data. partial_result reports cooperative interruption, not
    script cancellation; completed partial results may retain usable data.
    error is the failure diagnostic, otherwise None. Completed does not mean saved.
    """

    status: Literal["completed", "failed"]
    result: T | None
    error: str | None = None
    partial_result: bool = False


class Operation[T](Protocol):
    """Host-owned operation with a typed terminal delivery.

    The host starts and owns the work. Scripts wrap this handle in Wait rather
    than iterating it or accessing a runner. The host supplies delivery before
    advancing the suspended helper. This contract contains no runtime policy.
    """

    @property
    def name(self) -> str:
        """Return the operation name reported by the host, not analysis content."""
        raise NotImplementedError

    @property
    def delivery(self) -> OperationResult[T] | None:
        """Return None while pending, then this operation's terminal outcome."""
        raise NotImplementedError


class ScriptRequest:
    """Common yield type for Wait and Ask, without erasing their payload types.

    This marker is not an author extension point. Hosts accept only Wait and
    Ask requests and reject other values yielded by a script.
    """


@dataclass(frozen=True)
class Wait[T](ScriptRequest):
    """Suspend a script until op, a host-owned Operation[T], reaches terminal.

    op retains its typed OperationResult[T] delivery. Cancellation returns a
    cancelled outcome for author-owned cleanup; genuine failures remain failed
    outcomes and convenience helpers may raise the original operation error.
    Waiting does not expose analysis content to the agent.
    """

    op: Operation[T]


@dataclass
class Ask[A: Answer](ScriptRequest):
    """Suspend a script for an explicit agent or user decision.

    question contains authored evidence and declares the Answer model A.
    delivery is None until the host validates and supplies an A instance.
    Helpers return that instance after the host advances them with None.
    After a cancelled question resumes, delivery=None denotes cancellation.
    Asking never performs the action described by an accepted answer.
    """

    question: Question[A]
    delivery: A | None = None


type ScriptGenerator[R] = Generator[ScriptRequest, None, R]
"""Yield Wait/Ask, resume via request delivery plus None, return an authored R."""

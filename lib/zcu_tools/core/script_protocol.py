"""Stateless script outcomes and typed question/answer contracts.

These declarations perform no driving, I/O or source lookup. The async host
owns control scheduling and Future delivery for recipes and workflows.
"""

from typing import Literal

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

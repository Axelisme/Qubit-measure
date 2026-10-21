"""Caller registrations for native Ask hosting, with observable author deliveries."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from threading import get_ident

from pydantic import BaseModel, ConfigDict
from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.measure.services.workflow_attachment import WorkflowAttachment
from zcu_tools.gui.form import FormModel


class Values(FormModel):
    value: int = 3


class Native(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: int = 3


class Decision(Answer):
    value: int


class OtherAnswer(Answer):
    value: int


class Choose(Question[Decision]):
    # Deliberately collide with envelope metadata: authored fields stay nested.
    question_type: str = "authored evidence"
    point: int = 4


@dataclass
class AskAuthor:
    """Observe author code, not host internals; optionally hold cancellation cleanup."""

    deliveries: list[tuple[Decision | None, int]] = field(default_factory=list)
    captures: list[int] = field(default_factory=list)
    cancelled_followups: list[object] = field(default_factory=list)
    on_cancel: Callable[[], None] = lambda: None

    def attachment(
        self,
    ) -> WorkflowAttachment[Native, Native, int, int, Values, Values, None]:
        def init(env: InitEnv[None], plan: Native) -> int:
            return 0

        @workflow("asking", plan=Native, tunables=Native, init=init)
        def step(
            env: WorkflowEnv[None], plan: Native, tun: Native, state: int
        ) -> Step[int, int]:
            if state:
                return Done()
            self.captures.append(tun.value)
            answer = yield from env.ask(Choose())
            self.deliveries.append((answer, get_ident()))
            if answer is None:
                self.on_cancel()
                self.cancelled_followups.append(
                    (yield from env.wait_until(env.started_at + timedelta(days=1)))
                )
                self.cancelled_followups.append((yield from env.ask(Choose(point=99))))
                return Done()
            return Next(answer.value, 1)

        return WorkflowAttachment(
            step=step,
            plan_model=Values,
            knobs_model=Values,
            plan_to_cfg=lambda form, env, snapshot: Native(value=form.value),
            knobs_to_cfg=lambda form: Native(value=form.value),
            seed=lambda plan, form: None,
        )

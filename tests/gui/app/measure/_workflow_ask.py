"""Caller registrations for native Ask hosting, with observable author deliveries."""

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import get_ident

from pydantic import BaseModel, ConfigDict
from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.experiment.scope import ScriptCancelled
from zcu_tools.experiment.workflows import (
    Live,
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

    deliveries: list[tuple[Decision, int]] = field(default_factory=list)
    captures: list[int] = field(default_factory=list)
    on_cancel: Callable[[], None] = lambda: None
    finally_threads: list[int] = field(default_factory=list)
    on_finish: Callable[[], None] = lambda: None

    def attachment(
        self,
    ) -> WorkflowAttachment[Native, Native, None, Values, Values, None]:
        @workflow("asking", "Synthetic retained-worker Ask")
        async def script(
            env: WorkflowEnv[None], plan: Native, tun: Live[Native]
        ) -> None:
            self.captures.append(tun.current.value)
            try:
                answer = await env.ask(Choose())
                self.deliveries.append((answer, get_ident()))
                env.commit(Native(value=answer.value))
            except ScriptCancelled:
                self.on_cancel()
                raise
            finally:
                self.finally_threads.append(get_ident())
                self.on_finish()

        return WorkflowAttachment(
            script=script,
            plan_model=Values,
            knobs_model=Values,
            plan_to_cfg=lambda form, env, snapshot: Native(value=form.value),
            knobs_to_cfg=lambda form: Native(value=form.value),
            seed=lambda plan, form: None,
        )

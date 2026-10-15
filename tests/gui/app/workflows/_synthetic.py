"""Small independent workflow registrations for host contract tests."""

from dataclasses import dataclass
from typing import Annotated, Literal

from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from zcu_tools.gui.form import FormEnv, FormModel
from zcu_tools.gui.session.aliases import AliasRow

from tests.resources.entry.fakes import Connected, Sensor

Component = Connected
OtherComponent = Sensor

ALIAS_PRESET = (AliasRow("qubit", ""),)


@dataclass(frozen=True)
class Context:
    scale: float


class SampleForm(FormModel):
    value: float = Field(default=20.0, gt=0)
    count: int = Field(default=100, gt=0)
    factor: float = Field(default=4.0, gt=0)

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        data = super().defaults(env)
        if env.parameters is not None:
            try:
                env.parameters.value("general.ext.value")
            except (KeyError, ValueError, AttributeError):
                pass
            else:
                data["value"] = {"expr": "general.ext.value"}
        return data


class Tunables(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sample: SampleForm = Field(default_factory=SampleForm)


class TunablesForm(FormModel):
    sample: SampleForm = Field(default_factory=SampleForm)

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        data = super().defaults(env)
        data["sample"] = SampleForm.defaults(env)
        return data

    def to_cfg(self) -> Tunables:
        return Tunables(sample=self.sample)


class Listed(FormModel):
    kind: Literal["list"] = "list"
    values: list[float] = Field(default_factory=lambda: [1.0, 1.1])


class Linear(FormModel):
    kind: Literal["linrange"] = "linrange"
    start: float = 0.0
    stop: float = 1.0
    n: int = Field(default=3, gt=1)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snapshot: RunSnapshot
    points: int = Field(default=21, gt=0)
    samples: tuple[float, ...] = (1.0, 1.1)


class PlanForm(FormModel):
    scale: float = Field(gt=0)
    points: int = Field(default=21, gt=0)
    samples: Annotated[Listed | Linear, Field(discriminator="kind")] = Field(
        default_factory=Listed
    )

    @staticmethod
    def to_cfg(form: "PlanForm", _env: FormEnv, snapshot: RunSnapshot) -> Plan:
        if isinstance(form.samples, Listed):
            samples = tuple(form.samples.values)
        else:
            samples = tuple(
                form.samples.start
                + (form.samples.stop - form.samples.start) * i / (form.samples.n - 1)
                for i in range(form.samples.n)
            )
        return Plan(snapshot=snapshot, points=form.points, samples=samples)


def context_seed(_plan: Plan, form: PlanForm) -> Context:
    return Context(form.scale)


def init_sample(_env: InitEnv[Context], _plan: Plan) -> int:
    return 0


@workflow("sample", plan=Plan, tunables=Tunables, init=init_sample)
def sample_step(
    env: WorkflowEnv[Context], plan: Plan, tun: Tunables, state: int
) -> Step[int, float]:
    del plan
    if state:
        return Done()
    env.plots.subplots("live")

    def draw(figure: Figure) -> None:
        figure.subplots().plot([0, 1], [0, env.context.scale])

    env.plots.draw("live", draw)
    yield from ()
    return Next(tun.sample.value, 1)


def registration(step=sample_step):
    return WorkflowAttachment(
        step=step,
        plan_model=PlanForm,
        knobs_model=TunablesForm,
        plan_to_cfg=PlanForm.to_cfg,
        knobs_to_cfg=TunablesForm.to_cfg,
        seed=context_seed,
    )


def build_registry():
    return (registration(),)


def inline_input():
    from zcu_tools.gui.form import FormInput

    return FormInput(
        PlanForm,
        {"scale": 0.4, "points": 21, "samples": {"kind": "list", "values": [1.0, 1.1]}},
    )

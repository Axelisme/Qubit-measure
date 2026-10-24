"""Registered author shared by native pane and MCP figure-contract scenarios."""

import numpy as np
from zcu_tools import plotting
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.workflows import Failed, Live, WorkflowEnv, workflow
from zcu_tools.gui.app.measure.services.workflow_attachment import WorkflowAttachment

from ._workflow_ask import Choose, Native, Values


class PlotCfg(ConfigBase):
    stage: int


def acquire(cfg: PlotCfg, *, context: RunContext) -> int:
    viewer = plotting.live_line(
        context.plots, "measurement", "Delay", f"Attempt {cfg.stage}"
    )
    viewer.update(np.array([0.0, 1.0]), np.array([1.0, float(cfg.stage)]))
    if cfg.stage == 1:
        plotting.live_line(context.plots, "diagnostic", "Frequency", "Diagnostic")
    return cfg.stage


@workflow("operation-plots", "Replace operation figures without losing workflow trends")
async def script(env: WorkflowEnv[None], plan: Native, tun: Live[Native]) -> None:
    trend = plotting.live_line(env.plots, "trend", "Iteration", "Workflow trend")
    trend.update(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    for stage, iteration in zip((1, 2), env.iterations(), strict=False):
        outcome = await iteration.run(acquire, PlotCfg(stage=stage))
        if isinstance(outcome, Failed):
            raise RuntimeError(outcome.reason)
        iteration.commit(Native(value=outcome.value.result))
        await env.ask(Choose(point=stage))


def attachment() -> WorkflowAttachment[Native, Native, None, Values, Values, None]:
    """Supply ordinary forms and a core-shaped plotting operation, without hardware."""
    return WorkflowAttachment(
        script=script,
        plan_model=Values,
        knobs_model=Values,
        plan_to_cfg=lambda form, env, snapshot: Native(value=form.value),
        knobs_to_cfg=lambda form: Native(value=form.value),
        seed=lambda plan, form: None,
    )

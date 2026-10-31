"""Adapt core callables to explicit operation capabilities."""

from copy import deepcopy

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import JsonObject

from .acquisition import capture_device_snapshot
from .context import RunContext
from .operations import Completed, MissingCapability, RunFunction
from .records import RunRecord
from .scope import Scope
from .script_host import CallContext


def _validate_cfg(cfg: object) -> None:
    if not isinstance(cfg, ConfigBase):
        raise TypeError("run_core cfg must be a ConfigBase instance")


async def run_core[Cfg: ConfigBase, Result](
    scope: Scope,
    experiment: RunFunction[Cfg, Result],
    cfg: Cfg,
    *,
    name: str | None = None,
) -> Completed[RunRecord[Cfg, Result]]:
    """Call experiment(cfg, *, context) once under scope, without saving.

    scope must have a runtime supplying operation CallContext resources, as
    WorkflowEnv does. experiment may be a bound Experiment.run. cfg must be a
    ConfigBase; the helper deep-copies it before admission, leaving the caller's
    cfg unchanged while the core may mutate its owned copy.
    name is an optional explicit operation label. The returned RunRecord holds
    the core's returned value and cfg at return, with detached pre-call settings
    in metadata["device_snapshot"]. Every connected driver is read once before
    the core; empty devices records {}. Driver errors prevent the core call.
    Missing soc, soccfg, devices or plots raises MissingCapability; unavailable
    devices never falls back to an empty mapping. Non-ConfigBase cfg raises
    TypeError. Core and plotting errors propagate unchanged. The framework owns
    plots cleanup and completion facts. Child stops deliver the core's partial
    value; root stop raises ScriptCancelled after drain and fact recording.
    """
    _validate_cfg(cfg)
    captured = deepcopy(cfg)

    def invoke(ctx: CallContext) -> RunRecord[Cfg, Result]:
        if ctx.soc is None:
            raise MissingCapability("soc")
        if ctx.soccfg is None:
            raise MissingCapability("soccfg")
        if ctx.devices is None:
            raise MissingCapability("devices")
        if ctx.plots is None:
            raise MissingCapability("plots")
        context = RunContext(
            soc=ctx.soc,
            soccfg=ctx.soccfg,
            devices=ctx.devices,
            plots=ctx.plots,
            scope=ctx.scope,
        )
        snapshot: JsonObject = dict(capture_device_snapshot(context.devices))
        result = experiment(captured, context=context)
        return RunRecord(captured, result, metadata={"device_snapshot": snapshot})

    return await scope.call(invoke, name=name)

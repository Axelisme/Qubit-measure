"""Immutable value, generated-options and typed interactive declaration seams."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass

import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure import adapter
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.interactive import PluginDefinition, Session
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult


def saving_form(context: FormContext) -> Form[SavingCfg]:
    del context
    return Form(SavingCfg, {})


def saving_run(
    cfg: SavingCfg, *, context: RunContext
) -> RunRecord[SavingCfg, SavingResult]:
    return RunRecord(cfg=cfg, result=SavingCore().run(cfg, context=context))


@dataclass(frozen=True)
class Options:
    scale: float


@dataclass(frozen=True)
class Numeric(adapter.AnalyzeResultBase):
    value: float


def analyze(
    req: adapter.AnalyzeRequest[RunRecord[SavingCfg, SavingResult], Options],
    *,
    plots: Plots,
) -> Numeric:
    del plots
    return Numeric(
        float(req.run_result.result.signal.real.sum()) * req.analyze_params.scale
    )


def post_analyze(
    req: adapter.PostAnalyzeRequest[
        RunRecord[SavingCfg, SavingResult], Numeric, Options
    ],
    *,
    plots: Plots,
) -> Numeric:
    del plots
    return Numeric(req.analyze_result.value + req.post_analyze_params.scale)


def test_registry_publishes_an_immutable_value_and_rejects_invalid_registrations() -> (
    None
):
    definition = adapter.Adapter(
        spec=SavingCore.AXES_SPEC,
        make_form=saving_form,
        run=saving_run,
        guide=None,
        analysis=adapter.Analysis(fn=analyze, options=lambda source, env: Options(3.0)),
    )
    registry = Registry()
    registry.register("saving", definition)
    registry.validate()

    assert registry.create("saving") is definition
    assert definition.experiment_tag == "test/saving"
    assert definition.capabilities.analysis is adapter.AnalysisMode.FIT
    assert definition.capabilities.load_data
    assert definition.guide().resource_dir is None
    pytest.raises(TypeError, adapter.Adapter)
    pytest.raises(FrozenInstanceError, setattr, definition, "run", SavingCore().run)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("saving", definition)
    with pytest.raises(ValueError, match="Invalid adapter name"):
        registry.register("", definition)
    with pytest.raises(TypeError, match="Adapter value"):
        registry.register("invalid", object())


def test_generated_options_and_post_options_use_the_declared_source_and_primary() -> (
    None
):
    definition = adapter.Adapter(
        spec=SavingCore.AXES_SPEC,
        make_form=saving_form,
        run=saving_run,
        guide=None,
        analysis=adapter.Analysis(fn=analyze, options=lambda source, env: Options(3.0)),
        post_analysis=adapter.PostAnalysis(
            fn=post_analyze,
            options=lambda source, primary, env: Options(primary.value / 9.0),
        ),
    )
    cfg = SavingCfg()
    source = RunRecord(cfg, SavingCore().run(cfg, context=None))
    env = adapter.SessionEnv(parameters=None, soc=None, soccfg=None)
    options = definition.get_analyze_params(source, env)
    if options is None:
        pytest.fail("Declared options generator did not produce inputs")
    edited = adapter.reconstruct_params(type(options), {"scale": 4.0})
    plots = Plots(NonPresentingHost())
    try:
        primary = definition.analyze(
            adapter.AnalyzeRequest(source, edited, None), plots=plots
        )
        assert primary.value == 36.0
        post_options = definition.get_post_analyze_params(source, primary, env)
        if post_options is None:
            pytest.fail("Declared post options generator did not produce inputs")
        result = definition.post_analyze(
            adapter.PostAnalyzeRequest(source, primary, post_options, None),
            plots=plots,
        )
        assert result.value == 40.0
    finally:
        plots.finish()
        plots.release()


def test_interactive_uses_the_same_primary_slot_and_a_host_owned_typed_session() -> (
    None
):
    def plugin(
        req: adapter.AnalyzeRequest[RunRecord[SavingCfg, SavingResult], Options],
        *,
        plots: Plots,
    ) -> PluginDefinition[float, Numeric]:
        return PluginDefinition(
            plugin_id="synthetic",
            seed=req.analyze_params.scale,
            commands=(),
            can_finish=lambda state: None,
            build_result=Numeric,
        )

    def frontend(
        plugin: PluginDefinition[float, Numeric],
        session: Session[float],
        env: adapter.InteractiveFrontendEnv,
        finish: Callable[[], bool],
        cancel: Callable[[], bool],
        *,
        plots: Plots,
    ) -> object:
        return session.snapshot()

    definition = adapter.Adapter(
        spec=SavingCore.AXES_SPEC,
        make_form=saving_form,
        run=saving_run,
        guide=None,
        analysis=adapter.Interactive(
            plugin=plugin,
            frontend=frontend,
            options=lambda source, env: Options(3.0),
        ),
    )
    cleanup: list[Callable[[], None]] = []
    owner = ManualOwnerScheduler()

    def create_session[S, A](plugin: PluginDefinition[S, A]) -> Session[S]:
        session = plugin.open(owner)
        cleanup.append(session.dispose)
        return session

    cfg = SavingCfg()
    source = RunRecord(cfg, SavingCore().run(cfg, context=None))
    env = adapter.SessionEnv(parameters=None, soc=None, soccfg=None)
    plots = Plots(NonPresentingHost())
    try:
        mount = definition.make_interactive_frontend(
            adapter.AnalyzeRequest(source, Options(3.0), None),
            create_session,
            plots=plots,
        )
        assert definition.capabilities.analysis is adapter.AnalysisMode.INTERACTIVE
        assert mount(_FrontendEnv(), lambda: True, lambda: True) == 3.0
        assert definition.get_analyze_params(source, env) == Options(3.0)
    finally:
        for dispose in cleanup:
            dispose()
        plots.finish()
        plots.release()


class _FrontendEnv:
    def run_background(
        self,
        compute: Callable[[], object],
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        raise AssertionError("Contract construction must not submit background work")

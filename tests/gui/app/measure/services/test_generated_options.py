"""Caller-owned definitions connect native data, forms and analysis stages."""

from dataclasses import dataclass
from unittest.mock import MagicMock

from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure import adapter
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui.app.measure._form_fakes import form_resources, form_sources
from tests.gui.app.measure._pane_adapter import saving_form, saving_run


@dataclass(frozen=True)
class Options:
    scale: float


@dataclass(frozen=True)
class Numeric:
    value: float


def expect_options(value: object) -> Options:
    assert isinstance(value, Options)
    return value


def analyze(
    request: adapter.AnalyzeRequest[RunRecord[SavingCfg, SavingResult], Options],
    *,
    plots: Plots,
) -> Numeric:
    del plots
    return Numeric(
        float(request.run_result.result.signal.real.sum())
        * request.analyze_params.scale
    )


def test_generated_options_supply_schema_and_edits_until_source_replacement() -> None:
    generated: list[float] = []
    post_generated: list[float] = []

    def post_options(
        source: RunRecord[SavingCfg, SavingResult],
        primary: Numeric,
        env: adapter.SessionEnv,
    ) -> Options:
        assert env.parameters is None
        scale = float(source.result.signal.real.sum()) + primary.value
        post_generated.append(scale)
        return Options(scale)

    def post_analyze(
        request: adapter.PostAnalyzeRequest[
            RunRecord[SavingCfg, SavingResult], Numeric, Options
        ],
        *,
        plots: Plots,
    ) -> Numeric:
        return Numeric(request.analyze_result.value * request.post_analyze_params.scale)

    def options(
        source: RunRecord[SavingCfg, SavingResult], env: adapter.SessionEnv
    ) -> Options:
        assert env.parameters is None
        scale = float(source.result.signal.real.sum())
        generated.append(scale)
        return Options(scale)

    definition = adapter.Adapter(
        spec=SavingCore.AXES_SPEC,
        make_form=saving_form,
        run=saving_run,
        guide=None,
        analysis=adapter.Analysis(fn=analyze, options=options),
        post_analysis=adapter.PostAnalysis(fn=post_analyze, options=post_options),
    )
    registry = Registry()
    registry.register("saving", definition)
    state = State(adapter.SessionEnv(parameters=None, soc=None, soccfg=None))
    state.add_tab(
        "saving",
        Session(
            adapter_name="saving", adapter=registry.create("saving"), cfg=MagicMock()
        ),
    )
    service = TabService(
        state,
        registry,
        MagicMock(),
        form_resources(state),
        form_sources=lambda aliases: form_sources(state, aliases),
    )
    cfg = SavingCfg()
    source = RunRecord(cfg, SavingCore().run(cfg, context=None))
    state.update_tab_result("saving", source)
    assert service.prepare_result_analysis("saving").error is None
    for _ in range(2):
        definitions = service.analyze_param_definitions("saving", stage="primary")
        assert [field["name"] for field in definitions] == ["scale"]
    assert generated == [9.0]
    edited = expect_options(
        adapter.reconstruct_params(
            type(state.get_tab("saving").analysis.params), {"scale": 4.0}
        )
    )
    plots = Plots(NonPresentingHost())
    try:
        result = definition.analyze(
            adapter.AnalyzeRequest(source, edited, None), plots=plots
        )
        assert result.value == 36.0
        state.update_tab_analyze("saving", result, None, analyze_params_instance=edited)
        assert service.initialize_tab_post_analyze_params("saving") == Options(45.0)
        for _ in range(2):
            definitions = service.analyze_param_definitions("saving", stage="post")
            assert [field["name"] for field in definitions] == ["scale"]
        post_edited = expect_options(
            adapter.reconstruct_params(
                type(state.get_tab("saving").post_analysis.params), {"scale": 2.0}
            )
        )
        post_result = definition.post_analyze(
            adapter.PostAnalyzeRequest(source, result, post_edited, None), plots=plots
        )
        assert post_result.value == 72.0
        assert post_generated == [45.0]
    finally:
        plots.finish()
        plots.release()
    assert generated == [9.0]

    cfg = SavingCfg(gain=2.0)
    replacement = RunRecord(cfg, SavingCore().run(cfg, context=None))
    state.update_tab_result("saving", replacement)
    assert service.prepare_result_analysis("saving").error is None
    assert state.get_tab("saving").analysis.params == Options(18.0)
    assert generated == [9.0, 18.0]

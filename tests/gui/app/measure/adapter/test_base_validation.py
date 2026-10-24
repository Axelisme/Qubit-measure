from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.base import Experiment
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    PostAnalyzeResultBase,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import NonPresentingHost, Plots


class _MinimalNoAnalysisAdapter(BaseAdapter[Any, Any, None, None]):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE
    )
    exp_cls = Experiment


class _FitNoParamsAdapter(_MinimalNoAnalysisAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT
    )

    def analyze(self, req: Any, *, plots: Plots) -> None:
        del req
        return


@dataclass
class _RealParams:
    # All fields have defaults → base get_analyze_params can call _RealParams().
    flag: bool = False


@dataclass
class _RequiredParams:
    # No defaults → the base default cannot construct it; override is required.
    required_flag: bool


def test_fit_requires_analyze() -> None:
    with pytest.raises(TypeError, match="analysis=FIT.*analyze"):

        class _(_MinimalNoAnalysisAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.FIT
            )


def test_fit_forbids_interactive_plugin() -> None:
    with pytest.raises(TypeError, match="analysis=FIT.*make_interactive_plugin"):

        class _(_FitNoParamsAdapter):
            def make_interactive_plugin(self, req: Any, *, plots: Plots) -> Any:
                return object()


def test_fit_requires_analyze_params_hook_when_params_need_values() -> None:
    # Params without defaults cannot be constructed by the base default; the
    # validation must require an explicit get_analyze_params override.
    with pytest.raises(TypeError, match="params _RequiredParams.*get_analyze_params"):

        class _(BaseAdapter[Any, Any, None, _RequiredParams]):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.FIT
            )
            exp_cls = Experiment

            def analyze(self, req: Any, *, plots: Plots) -> None:
                del req
                return


def test_fit_allows_base_analyze_params_hook_when_params_all_have_defaults() -> None:
    # Params whose every field has a default are constructible by the base
    # default (params_cls()); no override is required.
    class _OkAdapter(BaseAdapter[Any, Any, None, _RealParams]):
        capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
            analysis=AnalysisMode.FIT
        )
        exp_cls = Experiment

        def analyze(self, req: Any, *, plots: Plots) -> None:
            del req
            return

    assert _OkAdapter.analyze_params_cls() is _RealParams
    # tolerated: Minimal contexts exercise the default params hook, which does not inspect either context.
    params = _OkAdapter().get_analyze_params(object(), object())  # pyright: ignore[reportArgumentType]
    assert isinstance(params, _RealParams)
    assert params.flag is False


def test_fit_allows_base_analyze_params_for_no_params() -> None:
    assert _FitNoParamsAdapter.analyze_params_cls() is type(None)


def test_interactive_requires_plugin_and_frontend() -> None:
    with pytest.raises(
        TypeError, match="analysis=INTERACTIVE.*make_interactive_plugin"
    ):

        class _(_MinimalNoAnalysisAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.INTERACTIVE
            )


def test_interactive_forbids_analyze() -> None:
    with pytest.raises(TypeError, match="analysis=INTERACTIVE.*analyze"):

        class _(_MinimalNoAnalysisAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.INTERACTIVE
            )

            def make_interactive_plugin(self, req: Any, *, plots: Plots) -> Any:
                return object()

            def make_interactive_frontend(
                self,
                plugin: Any,
                session: Any,
                env: Any,
                request_finish: Any,
                request_cancel: Any,
                *,
                plots: Plots,
            ) -> Any:
                return object()

            def analyze(self, req: Any, *, plots: Plots) -> None:
                del req
                return


def test_none_forbids_analyze() -> None:
    with pytest.raises(TypeError, match="analysis=NONE.*analyze"):

        class _(_MinimalNoAnalysisAdapter):
            def analyze(self, req: Any, *, plots: Plots) -> None:
                del req
                return


def test_none_forbids_interactive_plugin() -> None:
    with pytest.raises(TypeError, match="analysis=NONE.*make_interactive_plugin"):

        class _(_MinimalNoAnalysisAdapter):
            def make_interactive_plugin(self, req: Any, *, plots: Plots) -> Any:
                return object()


def test_interactive_requires_frontend() -> None:
    with pytest.raises(
        TypeError, match="analysis=INTERACTIVE.*make_interactive_frontend"
    ):

        class _(_MinimalNoAnalysisAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.INTERACTIVE
            )

            def make_interactive_plugin(self, req: Any, *, plots: Plots) -> Any:
                return object()


def test_none_forbids_get_analyze_params() -> None:
    with pytest.raises(TypeError, match="analysis=NONE.*get_analyze_params"):

        class _(_MinimalNoAnalysisAdapter):
            def get_analyze_params(self, result: Any, ctx: Any) -> None:
                del result, ctx
                return


def test_post_analysis_requires_fit() -> None:
    with pytest.raises(TypeError, match="post_analysis=True.*analysis=NONE"):

        class _(_MinimalNoAnalysisAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.NONE,
                post_analysis=True,
            )


def test_post_analysis_requires_hook_for_required_params() -> None:
    with pytest.raises(TypeError, match="post_analysis=True.*get_post_analyze_params"):

        class _(
            BaseAdapter[Any, Any, None, None, PostAnalyzeResultBase, _RequiredParams]
        ):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.FIT,
                post_analysis=True,
            )

            def analyze(self, req: Any, *, plots: Plots) -> None:
                return None

            def post_analyze(self, req: Any, *, plots: Plots) -> PostAnalyzeResultBase:
                del req
                return PostAnalyzeResultBase()


def test_post_analysis_requires_post_analyze() -> None:
    with pytest.raises(TypeError, match="post_analysis=True.*post_analyze"):

        class _(_FitNoParamsAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.FIT,
                post_analysis=True,
            )

            def get_post_analyze_params(self, analyze_result: Any, ctx: Any) -> None:
                del analyze_result, ctx
                return


def test_post_analysis_false_forbids_get_post_analyze_params() -> None:
    with pytest.raises(TypeError, match="post_analysis=False.*get_post_analyze_params"):

        class _(_FitNoParamsAdapter):
            def get_post_analyze_params(self, analyze_result: Any, ctx: Any) -> None:
                del analyze_result, ctx
                return


def test_post_analysis_false_forbids_post_analyze() -> None:
    with pytest.raises(TypeError, match="post_analysis=False.*post_analyze"):

        class _(_FitNoParamsAdapter):
            def post_analyze(self, req: Any, *, plots: Plots) -> None:
                del req
                return


def test_intermediate_base_implementation_counts_as_implemented() -> None:
    class _ChildAdapter(_FitNoParamsAdapter):
        pass

    assert _ChildAdapter.capabilities.analysis is AnalysisMode.FIT
    assert _ChildAdapter().analyze(object(), plots=Plots(NonPresentingHost())) is None


def test_intermediate_base_forbidden_implementation_is_detected() -> None:
    with pytest.raises(TypeError, match="analysis=NONE.*analyze"):

        class _(_FitNoParamsAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.NONE
            )


def test_base_run_preflight_default_is_noop() -> None:
    req: Any = object()
    cfg = ConfigBase()

    assert _MinimalNoAnalysisAdapter().validate_run_request(req, cfg) is None
    assert cfg.model_dump() == {}

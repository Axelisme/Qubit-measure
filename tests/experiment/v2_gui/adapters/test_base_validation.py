from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    NoAnalysisResult,
    NoAnalyzeParams,
    PostAnalyzeResultBase,
)
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
)


class _MinimalNoAnalysisAdapter(
    BaseAdapter[Any, Any, NoAnalysisResult, NoAnalyzeParams]
):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE
    )
    exp_cls: ClassVar[type[object]] = object

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec()

    def make_default_value(self, ctx: Any) -> CfgSectionValue:
        del ctx
        return CfgSectionValue()

    def make_filename_stem(self, ctx: Any) -> str:
        del ctx
        return "minimal"


class _FitNoParamsAdapter(_MinimalNoAnalysisAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT
    )

    def analyze(self, req: Any) -> NoAnalysisResult:
        del req
        return NoAnalysisResult()


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

        class _BadAdapter(_MinimalNoAnalysisAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.FIT
            )


def test_fit_forbids_interactive_setup() -> None:
    with pytest.raises(TypeError, match="analysis=FIT.*setup_interactive_analysis"):

        class _BadAdapter(_FitNoParamsAdapter):
            def setup_interactive_analysis(self, req: Any, host: Any) -> Any:
                del req, host
                return object()


def test_fit_requires_analyze_params_hook_when_params_need_values() -> None:
    # Params without defaults cannot be constructed by the base default; the
    # validation must require an explicit get_analyze_params override.
    with pytest.raises(TypeError, match="params _RequiredParams.*get_analyze_params"):

        class _BadAdapter(BaseAdapter[Any, Any, NoAnalysisResult, _RequiredParams]):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.FIT
            )
            exp_cls: ClassVar[type[object]] = object

            @classmethod
            def cfg_spec(cls) -> CfgSectionSpec:
                return CfgSectionSpec()

            def make_default_value(self, ctx: Any) -> CfgSectionValue:
                del ctx
                return CfgSectionValue()

            def make_filename_stem(self, ctx: Any) -> str:
                del ctx
                return "bad"

            def analyze(self, req: Any) -> NoAnalysisResult:
                del req
                return NoAnalysisResult()


def test_fit_allows_base_analyze_params_hook_when_params_all_have_defaults() -> None:
    # Params whose every field has a default are constructible by the base
    # default (params_cls()); no override is required.
    class _OkAdapter(BaseAdapter[Any, Any, NoAnalysisResult, _RealParams]):
        capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
            analysis=AnalysisMode.FIT
        )
        exp_cls: ClassVar[type[object]] = object

        @classmethod
        def cfg_spec(cls) -> CfgSectionSpec:
            return CfgSectionSpec()

        def make_default_value(self, ctx: Any) -> CfgSectionValue:
            del ctx
            return CfgSectionValue()

        def make_filename_stem(self, ctx: Any) -> str:
            del ctx
            return "ok"

        def analyze(self, req: Any) -> NoAnalysisResult:
            del req
            return NoAnalysisResult()

    assert _OkAdapter.analyze_params_cls() is _RealParams
    params = _OkAdapter().get_analyze_params(object(), object())  # type: ignore[arg-type]
    assert isinstance(params, _RealParams)
    assert params.flag is False


def test_fit_allows_base_analyze_params_for_no_params() -> None:
    assert _FitNoParamsAdapter.analyze_params_cls() is NoAnalyzeParams


def test_interactive_requires_setup() -> None:
    with pytest.raises(TypeError, match="analysis=INTERACTIVE.*setup_interactive"):

        class _BadAdapter(_MinimalNoAnalysisAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.INTERACTIVE
            )


def test_interactive_forbids_analyze() -> None:
    with pytest.raises(TypeError, match="analysis=INTERACTIVE.*analyze"):

        class _BadAdapter(_MinimalNoAnalysisAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.INTERACTIVE
            )

            def setup_interactive_analysis(self, req: Any, host: Any) -> Any:
                del req, host
                return object()

            def analyze(self, req: Any) -> NoAnalysisResult:
                del req
                return NoAnalysisResult()


def test_none_forbids_analyze() -> None:
    with pytest.raises(TypeError, match="analysis=NONE.*analyze"):

        class _BadAdapter(_MinimalNoAnalysisAdapter):
            def analyze(self, req: Any) -> NoAnalysisResult:
                del req
                return NoAnalysisResult()


def test_none_forbids_interactive_setup() -> None:
    with pytest.raises(TypeError, match="analysis=NONE.*setup_interactive"):

        class _BadAdapter(_MinimalNoAnalysisAdapter):
            def setup_interactive_analysis(self, req: Any, host: Any) -> Any:
                del req, host
                return object()


def test_none_forbids_get_analyze_params() -> None:
    with pytest.raises(TypeError, match="analysis=NONE.*get_analyze_params"):

        class _BadAdapter(_MinimalNoAnalysisAdapter):
            def get_analyze_params(self, result: Any, ctx: Any) -> NoAnalyzeParams:
                del result, ctx
                return NoAnalyzeParams()


def test_post_analysis_requires_fit() -> None:
    with pytest.raises(TypeError, match="post_analysis=True.*analysis=NONE"):

        class _BadAdapter(_MinimalNoAnalysisAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.NONE,
                post_analysis=True,
            )


def test_post_analysis_requires_get_post_analyze_params() -> None:
    with pytest.raises(TypeError, match="post_analysis=True.*get_post_analyze_params"):

        class _BadAdapter(_FitNoParamsAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.FIT,
                post_analysis=True,
            )

            def post_analyze(self, req: Any) -> PostAnalyzeResultBase:
                del req
                return PostAnalyzeResultBase()


def test_post_analysis_requires_post_analyze() -> None:
    with pytest.raises(TypeError, match="post_analysis=True.*post_analyze"):

        class _BadAdapter(_FitNoParamsAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.FIT,
                post_analysis=True,
            )

            def get_post_analyze_params(
                self, analyze_result: Any, ctx: Any
            ) -> NoAnalyzeParams:
                del analyze_result, ctx
                return NoAnalyzeParams()


def test_post_analysis_false_forbids_get_post_analyze_params() -> None:
    with pytest.raises(TypeError, match="post_analysis=False.*get_post_analyze_params"):

        class _BadAdapter(_FitNoParamsAdapter):
            def get_post_analyze_params(
                self, analyze_result: Any, ctx: Any
            ) -> NoAnalyzeParams:
                del analyze_result, ctx
                return NoAnalyzeParams()


def test_post_analysis_false_forbids_post_analyze() -> None:
    with pytest.raises(TypeError, match="post_analysis=False.*post_analyze"):

        class _BadAdapter(_FitNoParamsAdapter):
            def post_analyze(self, req: Any) -> PostAnalyzeResultBase:
                del req
                return PostAnalyzeResultBase()


def test_intermediate_base_implementation_counts_as_implemented() -> None:
    class _ChildAdapter(_FitNoParamsAdapter):
        pass

    assert _ChildAdapter.capabilities.analysis is AnalysisMode.FIT
    assert _ChildAdapter().analyze(object()).figure is None


def test_intermediate_base_forbidden_implementation_is_detected() -> None:
    with pytest.raises(TypeError, match="analysis=NONE.*analyze"):

        class _BadAdapter(_FitNoParamsAdapter):
            capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
                analysis=AnalysisMode.NONE
            )


def test_registered_adapters_import_with_capability_validation() -> None:
    from zcu_tools.experiment.v2_gui.registry import ADAPTERS

    assert "singleshot/t1_tone_sweep_gain" in ADAPTERS
    assert "singleshot/t1_tone_sweep_freq" in ADAPTERS

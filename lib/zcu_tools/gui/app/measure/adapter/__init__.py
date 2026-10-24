"""Measure GUI adapter contracts, request/result types, and lowering helpers."""

# Session-core value types are part of the adapter callback vocabulary, so
# authors can import them alongside the value declarations.
from zcu_tools.gui.app.measure.writeback import (
    ModuleWriteback,
    ParameterWriteback,
    SampleWriteback,
    WritebackItem,
)
from zcu_tools.gui.session.types import (
    ContextReadiness,
    SessionEnv,
    SocCfgHandle,
    SocCfgProtocol,
    SocHandle,
    SocProtocol,
)

from .analyze_params import ParamMeta, describe_analyze_params, reconstruct_params
from .hooks import InteractiveFrontendEnv
from .types import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    PostAnalyzeRequest,
    PostWritebackRequest,
    RunRequest,
    T_AnalyzeParams,
    T_AnalyzeResult,
    T_Cfg,
    T_PostAnalyzeParams,
    T_PostAnalyzeResult,
    T_Result,
    WritebackRequest,
)
from .validation import require_soc_handles
from .value import Adapter, Analysis, Interactive, PostAnalysis

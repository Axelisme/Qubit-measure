"""Measure GUI adapter contracts, request/result types, and lowering helpers."""

# Session-core value types live in ``gui/session/types``; re-exported here because
# ``ExpAdapterProtocol``'s signatures speak in them (the adapter contract's
# vocabulary), so adapter authors import them from the adapter package.
from zcu_tools.gui.session.types import (
    ContextReadiness,
    ExpContext,
    SocCfgHandle,
    SocCfgProtocol,
    SocHandle,
    SocProtocol,
)

from .analyze_params import ParamMeta, describe_analyze_params, reconstruct_params
from .protocol import ExpAdapterProtocol
from .types import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    AnalyzeResultWithFigure,
    ExperimentProtocol,
    InteractiveHost,
    InteractiveSession,
    LoadDataRequest,
    MetaDictWriteback,
    ModuleWriteback,
    NoAnalysisResult,
    NoAnalyzeParams,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    RunRequest,
    SaveDataRequest,
    SavePaths,
    T_AnalyzeParams,
    T_AnalyzeResult,
    T_Cfg,
    T_PostAnalyzeParams,
    T_PostAnalyzeResult,
    T_Result,
    WaveformWriteback,
    WritebackItem,
    WritebackRequest,
)
from .validation import require_soc_handles

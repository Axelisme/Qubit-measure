"""Measure GUI adapter contracts, request/result types, and lowering helpers."""

# Session-core value types live in ``gui/session/types``; re-exported here because
# ``ExpAdapterProtocol``'s signatures speak in them (the adapter contract's
# vocabulary), so adapter authors import them from the adapter package.
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
from .interactive import InteractivePluginProvider
from .protocol import ExpAdapterProtocol
from .types import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResult,
    AnalyzeResultBase,
    ExperimentProtocol,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
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

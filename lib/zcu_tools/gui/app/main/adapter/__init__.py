"""GUI adapter package — contracts, types, lowering, and inheritance helpers."""

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
from .inheritance import (
    align_locked_literals,
    inherit_from,
    make_default_value,
    select_ref_value_spec,
)
from .lowering import find_allowed_spec
from .protocol import ExpAdapterProtocol
from .types import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    AnalyzeResultWithFigure,
    CfgNodeSpec,
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    ExperimentProtocol,
    FloatSpec,
    InteractiveHost,
    InteractiveSession,
    IntSpec,
    LiteralSpec,
    LoadDataRequest,
    MetaDictWriteback,
    ModuleRefSpec,
    ModuleRefValue,
    ModuleWriteback,
    NoAnalysisResult,
    NoAnalyzeParams,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    RunRequest,
    SaveDataRequest,
    SavePaths,
    ScalarLeafInput,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    T_AnalyzeParams,
    T_AnalyzeResult,
    T_Cfg,
    T_PostAnalyzeParams,
    T_PostAnalyzeResult,
    T_Result,
    WaveformRefSpec,
    WaveformRefValue,
    WaveformWriteback,
    WritebackItem,
    WritebackRequest,
    default_value_for_type,
)
from .validation import require_soc_handles

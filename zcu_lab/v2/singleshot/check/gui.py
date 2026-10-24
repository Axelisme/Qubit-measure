from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.core.complex_codec import decode_complex
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import read_analysis_float
from zcu_lab.v2.singleshot.check.core import (
    CheckAnalyzeOptions,
    CheckCfg,
    CheckExp,
    CheckResult,
)
from zcu_lab.v2.singleshot.check.form import make_form

CheckRunResult: TypeAlias = RunRecord[CheckCfg, CheckResult]


def _require_complex(parameters: ParameterContext | None, path: str) -> complex:
    if parameters is None:
        raise RuntimeError(f"Analysis requires a captured parameter source: {path}")
    result = parameters.lookup(path)
    value = result.value if result.present else None
    if value is None:
        raise RuntimeError(
            f"Missing parameter calibration: {path}; apply singleshot/ge writeback"
        )
    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
        return complex(value)
    if isinstance(value, dict):
        try:
            return decode_complex(value)
        except (ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Parameter calibration is not a complex value: {path}"
            ) from exc
    raise RuntimeError(f"Parameter calibration is not a complex value: {path}")


def _read_ge_centers(
    parameters: ParameterContext | None,
) -> tuple[complex, complex, float]:
    """Read required canonical GE center/radius calibration from the Run capture.

    Centers accept real numbers, complex strings or the shared complex scalar
    encoding from the parameter document. Missing capture, missing calibration
    or invalid scalar encodings raise RuntimeError naming the canonical path.
    Invalid complex strings raise ValueError. Invalid aliases and unavailable
    sources propagate without fallback. The capture retains its raw YAML values.
    """
    g_center = _require_complex(parameters, "$qubit.ext.readout_cal.g_center")
    e_center = _require_complex(parameters, "$qubit.ext.readout_cal.e_center")
    radius = read_analysis_float(parameters, "$qubit.ext.readout_cal.radius")
    if radius is None:
        raise RuntimeError(
            "Missing parameter calibration: $qubit.ext.readout_cal.radius"
        )
    return g_center, e_center, radius


class CheckAdapter(BaseAdapter[CheckCfg, CheckRunResult]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[CheckCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = CheckExp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    # Standard run path (BaseAdapter.run): CheckExp.run(soc, soccfg, cfg) needs no
    # centres — the trio is an ANALYZE input, read from md in analyze().

    # No get_analyze_params override: None (4th generic arg).

    def run(
        self,
        req: RunRequest,
        cfg: CheckCfg,
        *,
        context: RunContext,
    ) -> CheckRunResult:
        del req
        result = CheckExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[CheckRunResult, None], *, plots: Plots
    ) -> None:
        # The check classifies the scatter against the GE centres — read the trio
        # from md (fast-fail if the upstream 'singleshot/ge' writeback is absent).
        g_center, e_center, radius = _read_ge_centers(req.parameters)
        CheckExp().analyze(
            req.run_result, CheckAnalyzeOptions(g_center, e_center, radius), plots=plots
        )
        return

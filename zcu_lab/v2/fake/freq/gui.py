"""freq experiment GUI attachment."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from zcu_tools.datafile import RunMetadata
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    LoadDataRequest,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SaveDataRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.fake.freq.core import (
    FakeFreqCfg,
    FakeFreqExp,
    FakeFreqRunResult,
    HangerSimParams,
    Param,
    TransmissionSimParams,
)
from zcu_lab.v2.fake.freq.form import FORM
from zcu_lab.v2.onetone.freq.core import FreqAnalyzeOptions


@dataclass
class FakeFreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    fwhm: float
    params: dict[str, Any]


@dataclass
class FakeFreqAnalyzeParams:
    model_type: Annotated[Literal["hm", "t", "auto"], ParamMeta(label="Model type")] = (
        "hm"
    )
    fit_bg_amp_slope: Annotated[bool, ParamMeta(label="Fit amplitude background")] = (
        False
    )
    fit_bg_phase_curvature: Annotated[bool, ParamMeta(label="Fit phase curvature")] = (
        False
    )


class FakeFreqAdapter(
    BaseAdapter[
        FakeFreqCfg,
        FakeFreqRunResult,
        FakeFreqAnalyzeResult,
        FakeFreqAnalyzeParams,
    ]
):
    """Simulated one-tone frequency sweep.  No hardware required."""

    experiment_tag: ClassVar[str | None] = FakeFreqExp.persistence_tag()

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False, load_data=True
    )
    exp_cls = FakeFreqExp
    ExpCfg_cls = FakeFreqCfg
    form = FORM

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Simulated one-tone resonator frequency sweep — a HangerModel "
            "lineshape plus Gaussian noise, computed in software with no "
            "hardware or SoC. Mirrors the real onetone/freq run/analyze/"
            "writeback flow so you can rehearse the analysis offline."
        ),
        expects_md=(
            "Uses the fixed resonator capture: resonator.freq is the sweep "
            "center and initial pulse/ADC frequency; resonator.kappa sets "
            "the half-span to five linewidths. resonator.wiring.ch/ro_ch "
            "supply generator/ADC channels. resonator.ext.time_of_fly plus "
            "0.05 us seeds trigger offset. Missing or null values leave "
            "required fields Invalid; no numeric fallback is substituted."
        ),
        expects_ml=(
            "Starts with a fresh inline pulse readout from the readout "
            "template and a const waveform. It does not adopt a calibrated "
            "readout slot or a named waveform. A library readout remains selectable."
        ),
        typical_writeback=(
            "Proposes the fitted resonator frequency and linewidth back into "
            "resonator.freq / resonator.kappa. The readout module / waveform are left "
            "to the user — a frequency fit alone does not justify rewriting "
            "the whole readout config."
        ),
        recommended=(
            "Analysis defaults to the hanger-model fit ('hm'). Switch to the "
            "transmission model ('t') and enable amplitude-background fitting "
            "when the signal-to-noise is poor or the magnitude baseline is "
            "visibly tilted. "
            "Use this adapter to validate an analysis pipeline before taking "
            "it to real hardware."
        ),
    )

    def __init__(
        self,
        model_type: Literal["t", "hm"] = "hm",
        params: Param | None = None,
        fast_mode: bool = False,
        persist_data: bool = True,
    ) -> None:
        if params is None:
            params = (
                HangerSimParams() if model_type == "hm" else TransmissionSimParams()
            )
        # Fast-Fail: the concrete params type must match model_type (strong types,
        # least surprise) — a hanger run with transmission params is a bug.
        expected = HangerSimParams if model_type == "hm" else TransmissionSimParams
        if not isinstance(params, expected):
            raise TypeError(
                f"model_type={model_type!r} expects {expected.__name__}, "
                f"got {type(params).__name__}"
            )
        self._model_type: Literal["t", "hm"] = model_type
        self._params: Param = params
        self._fast_mode = fast_mode
        # When True (default), save() writes a real (simulated-data) HDF5 to the
        # requested path — fake/freq is for rehearsing the full flow offline, so a
        # save should produce a file and "data saved to <path>" stays honest. Pass
        # False for a pure no-op (no file). See save().
        self._persist_data = persist_data

    def build_exp_cfg(
        self, raw_cfg: Mapping[str, ExperimentCfgValue], req: RunRequest
    ) -> FakeFreqCfg:
        return super().build_exp_cfg({**raw_cfg, "fast_mode": self._fast_mode}, req)

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> FakeFreqRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = FakeFreqExp(self._model_type, self._params).run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def load(self, req: LoadDataRequest) -> tuple[FakeFreqRunResult, RunMetadata]:
        """Read an exact native file with the configured core persistence spec.

        Return typed cfg/Result and original full history, without recapture or
        Labber fallback. Invalid native data raises ValueError; I/O errors
        propagate. The tab parameter capture does not replace file history.
        """
        spec = FakeFreqExp(self._model_type, self._params).persistence_spec()
        return load_run_with_metadata(Path(req.data_path), spec=spec)

    def analyze(
        self,
        req: AnalyzeRequest[FakeFreqRunResult, FakeFreqAnalyzeParams],
        *,
        plots: Plots,
    ) -> FakeFreqAnalyzeResult:
        analyze_params = req.analyze_params
        analysis = FakeFreqExp.analyze(
            req.run_result,
            FreqAnalyzeOptions(
                model_type=analyze_params.model_type,
                fit_bg_amp_slope=analyze_params.fit_bg_amp_slope,
                fit_bg_phase_curvature=analyze_params.fit_bg_phase_curvature,
            ),
            plots=plots,
        )
        return FakeFreqAnalyzeResult(
            freq=analysis.freq,
            fwhm=analysis.fwhm,
            params=analysis.params,
        )

    def get_writeback_items(
        self, req: WritebackRequest[FakeFreqRunResult, FakeFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            ParameterWriteback(
                target_name="r_f",
                description="Resonator frequency (MHz)",
                proposed_value=result.freq,
            ),
            ParameterWriteback(
                target_name="rf_w",
                description="Resonator linewidth FWHM (MHz)",
                proposed_value=result.fwhm,
            ),
        ]

    def save(self, req: SaveDataRequest[FakeFreqRunResult]) -> None:
        if not self._persist_data:
            return
        FakeFreqExp(self._model_type, self._params).save(
            req.run_result,
            Path(req.data_path),
            comment=req.comment or "fake/freq simulated data",
        )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.res_name}_freq_{time.strftime('%m%d')}"

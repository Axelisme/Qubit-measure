"""FakeFreqAdapter — simulates a one-tone frequency sweep using HangerModel.

FakeFreqExp mirrors the structure of FreqExp exactly (run_task + Task + LivePlot1D),
with a measure_fn that generates HangerModel signals plus Gaussian noise instead of
calling real hardware.  FakeFreqAdapter wraps FakeFreqExp and converts its flat
CfgSchema into the FakeFreqCfg that FakeFreqExp expects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    cast,
)

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.base import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.onetone.freq import (
    FreqCfg,
    FreqExp,
    FreqModuleCfg,
    FreqResult,
    FreqSweepCfg,
)
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_onetone_freq_writeback_items,
    make_pulse_module_spec,
    make_readout_default,
    make_readout_module_spec,
    make_reset_module_spec,
    md_get_float,
)
from zcu_tools.gui.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    ParamMeta,
    RunRequest,
    SaveDataRequest,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    AbsReadoutCfg,
    ProgramV2Cfg,
    PulseCfg,
    PulseReadoutCfg,
    ResetCfg,
)
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.utils.fitting.resonance.hanger import HangerModel

# ---------------------------------------------------------------------------
# FakeFreqCfg — same structure as FreqCfg but with HangerModel params
# ---------------------------------------------------------------------------


class FakeFreqSweepCfg(ProgramV2Cfg):  # type: ignore[misc]
    freq: SweepCfg


class FakeFreqModelCfg(ProgramV2Cfg):  # type: ignore[misc]
    freq: float = 6000.0
    Ql: float = 5000.0
    Qc_abs: float = 6000.0
    phi: float = 0.0
    a0_abs: float = 1.0
    edelay: float = 0.05
    noise_scale: float = 0.05


class FakeFreqModuleCfg(ConfigBase):
    readout: AbsReadoutCfg
    init_pulse: Optional[PulseCfg] = None
    reset: Optional[ResetCfg] = None


class FakeFreqCfg(ProgramV2Cfg, ExpCfgModel):
    sweep: FakeFreqSweepCfg
    model: FakeFreqModelCfg = FakeFreqModelCfg()
    modules: FakeFreqModuleCfg
    fast_mode: bool = False  # skip per-point sleep; set True in tests


# ---------------------------------------------------------------------------
# Result types — named dataclasses, no side channels
# ---------------------------------------------------------------------------


FakeFreqRunResult: TypeAlias = FreqResult


@dataclass
class FakeFreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    fwhm: float
    params: dict[str, Any]
    figure: Figure


@dataclass
class FakeFreqAnalyzeParams:
    model_type: Annotated[Literal["hm", "t", "auto"], ParamMeta(label="Model type")]
    fit_bg_slope: Annotated[bool, ParamMeta(label="Fit background slope")]


# ---------------------------------------------------------------------------
# FakeFreqExp — same run structure as FreqExp, fake measure_fn
# ---------------------------------------------------------------------------


class FakeFreqExp(AbsExperiment[FreqResult, FakeFreqCfg]):
    """Simulated FreqExp: same run/analyze/save interface, no hardware required."""

    def run(self, cfg: FakeFreqCfg) -> FreqResult:
        sweep = cfg.sweep.freq
        freqs = np.linspace(sweep.start, sweep.stop, sweep.expts)

        m = cfg.model
        a0 = complex(m.a0_abs)
        Qc = complex(m.Qc_abs * np.exp(-1j * m.phi))
        clean = HangerModel.calc_signals(
            freqs, m.freq, m.Ql, cast(float, Qc), m.phi, a0, m.edelay
        )
        sigma = m.noise_scale / np.sqrt(cfg.reps * cfg.rounds)
        rng = np.random.default_rng()

        def measure_fn(
            ctx: TaskState,
            update_hook: Optional[Callable[[int, NDArray[np.complex128]], None]],
        ) -> NDArray[np.complex128]:
            accumulated = np.zeros(len(freqs), dtype=np.complex128)
            rounds_done = 0
            for r in range(cfg.rounds):
                if ctx.is_stop():
                    break
                noise = rng.normal(0, sigma * np.sqrt(cfg.rounds), len(freqs))
                noise_i = rng.normal(0, sigma * np.sqrt(cfg.rounds), len(freqs))
                accumulated += clean + noise + 1j * noise_i
                rounds_done += 1
                if not cfg.fast_mode:
                    for _ in range(len(freqs)):
                        time.sleep(0.0005)
                if update_hook is not None:
                    update_hook(r + 1, accumulated / rounds_done)
            return accumulated / max(rounds_done, 1)

        with LivePlot1D("Frequency (MHz)", "Amplitude", auto_close=False) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw,
                    result_shape=(len(freqs),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(freqs, np.abs(ctx.root_data)),
            )

        return FreqResult(freqs=freqs, signals=signals)

    def analyze(
        self,
        result: Optional[FreqResult] = None,
        *,
        model_type: Literal["hm", "t", "auto"] = "auto",
        fit_bg_slope: bool = False,
    ) -> tuple[float, float, dict[str, Any], Figure]:
        assert result is not None
        return FreqExp().analyze(
            result, model_type=model_type, fit_bg_slope=fit_bg_slope
        )

    def save(self, result: FreqResult) -> None:
        pass  # no real hardware, skip HDF5 persistence


# ---------------------------------------------------------------------------
# FakeFreqAdapter — wraps FakeFreqExp, converts CfgSchema → FakeFreqCfg
# ---------------------------------------------------------------------------


class FakeFreqAdapter(
    BaseAdapter[
        FakeFreqCfg,
        FakeFreqRunResult,
        FakeFreqAnalyzeResult,
        FakeFreqAnalyzeParams,
    ]
):
    """Simulated one-tone frequency sweep.  No hardware required."""

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False
    )
    exp_cls = FakeFreqExp

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Simulated one-tone resonator frequency sweep — a HangerModel "
                "lineshape plus Gaussian noise, computed in software with no "
                "hardware or SoC. Mirrors the real onetone/freq run/analyze/"
                "writeback flow so you can rehearse the analysis offline."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional): 'r_f' — resonator "
                "frequency, the sweep centre (~4000–8000 MHz); 'rf_w' — linewidth, "
                "used to set the sweep span and a loaded-Q guess (~0.1–5 MHz); "
                "'res_ch' / 'ro_ch' — drive / readout channel indices; 'timeFly' "
                "— cable time-of-flight feeding the trigger offset (~0–1 us)."
            ),
            expects_ml=(
                "Needs a readout module to shape the probe pulse, and references a "
                "ModuleLibrary waveform named 'ro_waveform' when one exists "
                "(optional)."
            ),
            typical_writeback=(
                "Proposes the fitted resonator frequency and linewidth back into "
                "MetaDict 'r_f' / 'rf_w', and an updated 'readout_rf' module + "
                "'ro_waveform' waveform into the ModuleLibrary."
            ),
            recommended=(
                "Analysis defaults to the hanger-model fit ('hm'). Switch to the "
                "transmission model ('t') and enable background-slope fitting when "
                "the signal-to-noise is poor or the baseline is visibly tilted. "
                "Use this adapter to validate an analysis pipeline before taking "
                "it to real hardware."
            ),
        )

    def __init__(self, fast_mode: bool = False) -> None:
        self._fast_mode = fast_mode

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "readout": make_readout_module_spec(),
                        "init_pulse": make_pulse_module_spec(optional=True),
                        "reset": make_reset_module_spec(optional=True),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={"freq": SweepSpec(label="Freq (MHz)")},
                ),
                "model": CfgSectionSpec(
                    label="Model",
                    fields={
                        "freq": ScalarSpec(
                            label="Resonator freq (MHz)", type=float, decimals=2
                        ),
                        "Ql": ScalarSpec(label="Ql (loaded Q)", type=int),
                        "Qc_abs": ScalarSpec(label="|Qc| (coupling Q)", type=int),
                        "phi": ScalarSpec(label="phi (rad)", type=float, decimals=4),
                        "a0_abs": ScalarSpec(
                            label="|a0| (bg amplitude)", type=float, decimals=4
                        ),
                        "edelay": ScalarSpec(
                            label="edelay (us)", type=float, decimals=3
                        ),
                        "noise_scale": ScalarSpec(
                            label="Noise scale", type=float, decimals=4
                        ),
                    },
                ),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        r_f = md_get_float(ctx, "r_f", 6000.0)
        rf_w_raw = ctx.md.get("rf_w")
        rf_w: Optional[float] = (
            float(rf_w_raw) if isinstance(rf_w_raw, (int, float)) else None
        )

        # Sweep range: ±5× linewidth around r_f, or ±200 MHz if rf_w unknown
        half_span = (rf_w * 5.0) if rf_w is not None else 200.0
        freq_start = r_f - half_span
        freq_stop = r_f + half_span

        # Rough Ql estimate from linewidth: Ql ≈ r_f / rf_w
        ql_default = round(r_f / rf_w) if rf_w is not None and rf_w > 0 else 5000
        qc_default = ql_default * 2

        return CfgSectionValue(
            fields={
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
                "sweep": CfgSectionValue(
                    fields={
                        "freq": SweepValue(start=freq_start, stop=freq_stop, expts=201)
                    }
                ),
                "model": CfgSectionValue(
                    fields={
                        "freq": DirectValue(r_f),
                        "Ql": DirectValue(ql_default),
                        "Qc_abs": DirectValue(qc_default),
                        "phi": DirectValue(0.0),
                        "a0_abs": DirectValue(1.0),
                        "edelay": DirectValue(0.05),
                        "noise_scale": DirectValue(0.05),
                    }
                ),
                "modules": CfgSectionValue(
                    fields={
                        "readout": make_readout_default(ctx),
                        # init_pulse and reset are optional ModuleRefs; omitting their
                        # keys here means they start as disabled (None) in the UI.
                    }
                ),
            }
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FakeFreqCfg:
        return req.ml.make_cfg(raw_cfg, FakeFreqCfg, fast_mode=self._fast_mode)

    def run(self, req: RunRequest, schema: CfgSchema) -> FakeFreqRunResult:
        import dataclasses

        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = FakeFreqExp().run(cfg)
        freq_cfg = FreqCfg(
            reps=cfg.reps,
            rounds=cfg.rounds,
            relax_delay=cfg.relax_delay,
            modules=FreqModuleCfg(
                readout=cast(PulseReadoutCfg, cfg.modules.readout),
                reset=cfg.modules.reset,
            ),
            sweep=FreqSweepCfg(freq=cfg.sweep.freq),
        )
        return dataclasses.replace(result, cfg_snapshot=freq_cfg)

    def get_analyze_params(
        self, result: FakeFreqRunResult, ctx: ExpContext
    ) -> FakeFreqAnalyzeParams:
        return FakeFreqAnalyzeParams(model_type="hm", fit_bg_slope=False)

    def analyze(
        self,
        req: AnalyzeRequest[FakeFreqRunResult, FakeFreqAnalyzeParams],
    ) -> FakeFreqAnalyzeResult:
        analyze_params = req.analyze_params
        freq, fwhm, fit_params, figure = FakeFreqExp().analyze(
            req.run_result,
            model_type=analyze_params.model_type,
            fit_bg_slope=analyze_params.fit_bg_slope,
        )
        return FakeFreqAnalyzeResult(
            freq=freq,
            fwhm=fwhm,
            params=fit_params,
            figure=figure,
        )

    def get_writeback_items(
        self, req: WritebackRequest[FakeFreqRunResult, FakeFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        cfg = req.run_result.cfg_snapshot
        assert cfg is not None, "cfg_snapshot is required for writeback"
        return make_onetone_freq_writeback_items(
            cfg.modules.readout, result.freq, result.fwhm, req.ctx
        )

    def save(self, req: SaveDataRequest[FakeFreqRunResult]) -> None:
        del req  # fake experiment — no HDF5 persistence

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_freq_{time.strftime('%m%d')}"

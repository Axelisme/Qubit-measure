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
    CfgBuilder,
    make_onetone_freq_writeback_items,
    make_readout_module_spec,
    md_get_float,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    ParamMeta,
    RunRequest,
    SaveDataRequest,
    ScalarSpec,
    SweepSpec,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    AbsReadoutCfg,
    ProgramV2Cfg,
    PulseReadoutCfg,
)
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.utils.fitting.resonance.hanger import HangerModel
from zcu_tools.utils.fitting.resonance.transmission import TransmissionModel

# ---------------------------------------------------------------------------
# FakeFreqCfg — same structure as FreqCfg but with HangerModel params
# ---------------------------------------------------------------------------


class FakeFreqSweepCfg(ProgramV2Cfg):  # type: ignore[misc]
    freq: SweepCfg


# ---------------------------------------------------------------------------
# Simulation params — the ground-truth resonance the fake places, supplied at
# adapter construction (NOT in the cfg). Keeping them out of the cfg is the
# point: the sweep is set independently (from r_f/rf_w), so the analysis must
# genuinely *find* the dip rather than read an aligned cfg field. One frozen
# dataclass per resonator model, mirroring each model's calc_signals inputs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HangerSimParams:
    """Ground-truth params for a HangerModel lineshape (hanger / notch)."""

    freq: float = 6000.0
    Ql: float = 5000.0
    Qc_abs: float = 6000.0
    phi: float = 0.0
    a0_abs: float = 1.0
    edelay: float = 0.05
    noise_scale: float = 0.05


@dataclass(frozen=True)
class TransmissionSimParams:
    """Ground-truth params for a TransmissionModel lineshape (no Qc / phi)."""

    freq: float = 6000.0
    Ql: float = 5000.0
    a0_abs: float = 1.0
    edelay: float = 0.05
    noise_scale: float = 0.05


Param: TypeAlias = "HangerSimParams | TransmissionSimParams"


class FakeFreqModuleCfg(ConfigBase):
    # Mirrors the real onetone ExpCfg modules: readout only. No init_pulse (no
    # qubit-drive pulse) and no reset (one-tone runs without a qubit reset).
    readout: AbsReadoutCfg


class FakeFreqCfg(ProgramV2Cfg, ExpCfgModel):
    sweep: FakeFreqSweepCfg
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
    """Simulated FreqExp: same run/analyze/save interface, no hardware required.

    The ground-truth resonance (``model_type`` + ``params``) is supplied at
    construction, NOT carried in the cfg — so the cfg's sweep is set
    independently and the analysis must genuinely find the dip.
    """

    def __init__(self, model_type: Literal["t", "hm"], params: Param) -> None:
        self._model_type = model_type
        self._params = params

    def _clean_signals(self, freqs: NDArray[np.float64]) -> NDArray[np.complex128]:
        p = self._params
        a0 = complex(p.a0_abs)
        if self._model_type == "hm":
            assert isinstance(p, HangerSimParams)
            Qc = complex(p.Qc_abs * np.exp(-1j * p.phi))
            return HangerModel.calc_signals(
                freqs, p.freq, p.Ql, cast(float, Qc), p.phi, a0, p.edelay
            )
        assert isinstance(p, TransmissionSimParams)
        return TransmissionModel.calc_signals(freqs, p.freq, p.Ql, a0, p.edelay)

    def run(self, cfg: FakeFreqCfg) -> FreqResult:
        sweep = cfg.sweep.freq
        freqs = np.linspace(sweep.start, sweep.stop, sweep.expts)

        clean = self._clean_signals(freqs)
        sigma = self._params.noise_scale / np.sqrt(cfg.reps * cfg.rounds)
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

    @staticmethod
    def analyze(
        result: Optional[FreqResult] = None,
        *,
        model_type: Literal["hm", "t", "auto"] = "auto",
        fit_bg_slope: bool = False,
    ) -> tuple[float, float, dict[str, Any], Figure]:
        # Analysis is blind by construction — it only sees the result, never the
        # ground-truth sim params; no instance state needed.
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
                "MetaDict 'r_f' / 'rf_w'. The readout module / waveform are left "
                "to the user — a frequency fit alone does not justify rewriting "
                "the whole readout config."
            ),
            recommended=(
                "Analysis defaults to the hanger-model fit ('hm'). Switch to the "
                "transmission model ('t') and enable background-slope fitting when "
                "the signal-to-noise is poor or the baseline is visibly tilted. "
                "Use this adapter to validate an analysis pipeline before taking "
                "it to real hardware."
            ),
        )

    def __init__(
        self,
        model_type: Literal["t", "hm"] = "hm",
        params: Optional[Param] = None,
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

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        # Mirrors onetone/freq: the freq sweep owns the readout
                        # frequency, so lock it off the form (the sim reads the
                        # sweep range directly and ignores this field anyway).
                        "readout": make_readout_module_spec()
                        .lock_literal("pulse_cfg.freq", 0.0)
                        .lock_literal("ro_cfg.ro_freq", 0.0),
                    },
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={"freq": SweepSpec(label="Freq (MHz)")},
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                # No 'model' block: the simulated resonance is fixed at adapter
                # construction (model_type + params), hidden from the cfg, so the
                # sweep below scans blind and the analysis must find the dip.
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        r_f = md_get_float(ctx, "r_f", 6000.0)
        rf_w_raw = ctx.md.get("rf_w")
        rf_w: Optional[float] = (
            float(rf_w_raw) if isinstance(rf_w_raw, (int, float)) else None
        )

        # Sweep range: ±5× linewidth around r_f, or ±200 MHz if rf_w unknown.
        # This is set from r_f independently of the simulated resonance freq
        # (held in __init__); the two only coincide by default, so a test that
        # constructs a different params.freq forces a genuine blind sweep.
        half_span = (rf_w * 5.0) if rf_w is not None else 200.0
        freq_start = r_f - half_span
        freq_stop = r_f + half_span

        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=100)
            .role("modules.readout", "readout", prefer_blank=True)
            .sweep("sweep.freq", freq_start, freq_stop, 201)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FakeFreqCfg:
        return req.ml.make_cfg(raw_cfg, FakeFreqCfg, fast_mode=self._fast_mode)

    def run(self, req: RunRequest, schema: CfgSchema) -> FakeFreqRunResult:
        import dataclasses

        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = FakeFreqExp(self._model_type, self._params).run(cfg)
        freq_cfg = FreqCfg(
            reps=cfg.reps,
            rounds=cfg.rounds,
            relax_delay=cfg.relax_delay,
            modules=FreqModuleCfg(
                readout=cast(PulseReadoutCfg, cfg.modules.readout),
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
        freq, fwhm, fit_params, figure = FakeFreqExp.analyze(
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
        return make_onetone_freq_writeback_items(result.freq, result.fwhm)

    def save(self, req: SaveDataRequest[FakeFreqRunResult]) -> None:
        # Pure-mock default: no HDF5 (no real instrument data). When the adapter
        # was built with persist_data=True, write the simulated sweep so the
        # "data saved to <path>" report is truthful and the file exists.
        if not self._persist_data:
            return
        from zcu_tools.utils.datasaver import save_data

        result = req.run_result
        save_data(
            filepath=req.data_path,
            x_info={"name": "Frequency", "unit": "Hz", "values": result.freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": result.signals},
            comment=req.comment or "fake/freq simulated data",
            tag="fake/freq",
        )

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_freq_{time.strftime('%m%d')}"

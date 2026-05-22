"""FakeFreqAdapter — simulates a one-tone frequency sweep using HangerModel.

FakeFreqExp mirrors the structure of FreqExp exactly (run_task + Task + LivePlot1D),
with a measure_fn that generates HangerModel signals plus Gaussian noise instead of
calling real hardware.  FakeFreqAdapter wraps FakeFreqExp and converts its flat
CfgSchema into the FakeFreqCfg that FakeFreqExp expects.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, cast

logger = logging.getLogger(__name__)

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Callable

from zcu_tools.experiment.base import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.onetone.freq import FreqExp
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2_gui.adapters.shared import (
    build_readout_for_frequency,
    build_waveform_for_length,
    make_flat_top_waveform_edit_template,
    make_module_ref_default,
    make_readout_edit_template,
)
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    ModuleRefSpec,
    ModuleWriteback,
    ParamSpec,
    SavePaths,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    WaveformWriteback,
    schema_to_dict,
)
from zcu_tools.gui.specs.readout import (
    make_direct_readout_spec,
    make_pulse_readout_spec,
)
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    AbsReadoutCfg,
    ModuleCfgFactory,
    ProgramV2Cfg,
)
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.utils.datasaver import create_datafolder
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


class FakeFreqCfg(ProgramV2Cfg, ExpCfgModel):
    sweep: FakeFreqSweepCfg
    model: FakeFreqModelCfg = FakeFreqModelCfg()
    modules: dict[str, Any] = {}
    fast_mode: bool = False  # skip per-point sleep; set True in tests


# ---------------------------------------------------------------------------
# Result types — named dataclasses, no side channels
# ---------------------------------------------------------------------------


@dataclass
class FreqRunResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: FakeFreqCfg  # immutable record of the run configuration


@dataclass
class FakeFreqAnalyzeResult:
    freq: float
    fwhm: float
    params: dict[str, Any]
    figure: Figure
    run_result: FreqRunResult  # back-reference for writeback


# ---------------------------------------------------------------------------
# FakeFreqExp — same run structure as FreqExp, fake measure_fn
# ---------------------------------------------------------------------------


class FakeFreqExp(AbsExperiment[FreqRunResult, FakeFreqCfg]):
    """Simulated FreqExp: same run/analyze/save interface, no hardware required."""

    def run(self, cfg: FakeFreqCfg) -> FreqRunResult:
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

        return FreqRunResult(freqs=freqs, signals=signals, cfg_snapshot=cfg)

    def analyze(
        self,
        result: Optional[FreqRunResult] = None,
        *,
        model_type: Literal["hm", "t", "auto"] = "auto",
        fit_bg_slope: bool = False,
    ) -> tuple[float, float, dict[str, Any], Figure]:
        assert result is not None
        raw_result = (result.freqs, result.signals)
        return FreqExp().analyze(
            raw_result, model_type=model_type, fit_bg_slope=fit_bg_slope
        )


# ---------------------------------------------------------------------------
# FakeFreqAdapter — wraps FakeFreqExp, converts CfgSchema → FakeFreqCfg
# ---------------------------------------------------------------------------


class FakeFreqAdapter(AbsExpAdapter[FreqRunResult, FakeFreqAnalyzeResult]):
    """Simulated one-tone frequency sweep.  No hardware required."""

    def __init__(self, fast_mode: bool = False) -> None:
        self._fast_mode = fast_mode

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:  # noqa: ARG002
        r_f: float = 6000.0
        rf_w: Optional[float] = None
        _r_f = getattr(ctx.md, "r_f", None)
        if isinstance(_r_f, (int, float)):
            r_f = float(_r_f)
        _rf_w = getattr(ctx.md, "rf_w", None)
        if isinstance(_rf_w, (int, float)):
            rf_w = float(_rf_w)

        # Sweep range: ±5× linewidth around r_f, or ±200 MHz if rf_w unknown
        half_span = (rf_w * 5.0) if rf_w is not None else 200.0
        freq_start = r_f - half_span
        freq_stop = r_f + half_span

        # Rough Ql estimate from linewidth: Ql ≈ r_f / rf_w
        ql_default = round(r_f / rf_w) if rf_w is not None and rf_w > 0 else 5000
        qc_default = ql_default * 2

        root_spec = CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    collapsible=True,
                    fields={
                        "readout": ModuleRefSpec(
                            allowed=[
                                make_direct_readout_spec(),
                                make_pulse_readout_spec(),
                            ],
                            label="Readout",
                        )
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "freq": SweepSpec(label="Freq (MHz)"),
                "res_freq": ScalarSpec(
                    label="Resonator freq (MHz)", type=float, decimals=2
                ),
                "Ql": ScalarSpec(label="Ql (loaded Q)", type=int),
                "Qc_abs": ScalarSpec(label="|Qc| (coupling Q)", type=int),
                "phi": ScalarSpec(label="phi (rad)", type=float, decimals=4),
                "a0_abs": ScalarSpec(
                    label="|a0| (bg amplitude)", type=float, decimals=4
                ),
                "edelay": ScalarSpec(label="edelay (us)", type=float, decimals=3),
                "noise_scale": ScalarSpec(label="Noise scale", type=float, decimals=4),
            }
        )

        root_val = CfgSectionValue(
            fields={
                "reps": ScalarValue(100),
                "rounds": ScalarValue(100),
                "freq": SweepValue(start=freq_start, stop=freq_stop, expts=201),
                "res_freq": ScalarValue(r_f),
                "Ql": ScalarValue(ql_default),
                "Qc_abs": ScalarValue(qc_default),
                "phi": ScalarValue(0.0),
                "a0_abs": ScalarValue(1.0),
                "edelay": ScalarValue(0.05),
                "noise_scale": ScalarValue(0.05),
                "modules": CfgSectionValue(
                    fields={
                        "readout": make_module_ref_default(
                            ml=ctx.ml,
                            module_type=AbsReadoutCfg,
                            preferred_names=["readout_rf", "readout", "res_readout"],
                        ),
                    }
                ),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def _schema_to_exp_cfg(self, schema: CfgSchema, ctx: ExpContext) -> FakeFreqCfg:
        d = schema_to_dict(schema, ctx.ml)
        sweep_cfg: SweepCfg = d["freq"]  # SweepCfg from make_sweep via SweepValue

        # Convert raw dicts to ModuleCfg objects for better writeback support
        modules_raw = d.get("modules", {})

        def _is_direct_readout_complete(raw: dict) -> bool:
            return all(key in raw for key in ("ro_ch", "ro_length", "ro_freq"))

        def _is_pulse_cfg_complete(raw: dict) -> bool:
            if not all(key in raw for key in ("waveform", "ch", "nqz", "freq", "gain")):
                return False
            waveform = raw.get("waveform")
            return isinstance(waveform, dict) and "style" in waveform

        def _should_convert_module(raw: dict) -> bool:
            type_val = raw.get("type")
            if type_val == "readout/direct":
                return _is_direct_readout_complete(raw)
            if type_val == "readout/pulse":
                pulse_cfg = raw.get("pulse_cfg")
                ro_cfg = raw.get("ro_cfg")
                return (
                    isinstance(pulse_cfg, dict)
                    and isinstance(ro_cfg, dict)
                    and _is_pulse_cfg_complete(pulse_cfg)
                    and _is_direct_readout_complete(ro_cfg)
                )
            return True

        modules = {}
        for k, v in modules_raw.items():
            try:
                if isinstance(v, dict) and "type" in v:
                    if _should_convert_module(v):
                        modules[k] = ModuleCfgFactory.from_raw(v, ml=ctx.ml)
                    else:
                        modules[k] = v
                else:
                    modules[k] = v
            except Exception as e:
                logger.warning("Failed to convert module %r to object: %s", k, e)
                modules[k] = v

        return FakeFreqCfg(
            reps=int(d["reps"]),
            rounds=int(d["rounds"]),
            sweep=FakeFreqSweepCfg(freq=sweep_cfg),
            model=FakeFreqModelCfg(
                freq=float(d["res_freq"]),
                Ql=float(d["Ql"]),
                Qc_abs=float(d["Qc_abs"]),
                phi=float(d["phi"]),
                a0_abs=float(d["a0_abs"]),
                edelay=float(d["edelay"]),
                noise_scale=float(d["noise_scale"]),
            ),
            modules=modules,
            fast_mode=self._fast_mode,
        )

    def get_run_params(self) -> dict[str, ParamSpec]:
        return {}

    def run(
        self,
        ctx: ExpContext,
        schema: CfgSchema,
        **user_params: Any,  # noqa: ARG002
    ) -> FreqRunResult:
        cfg = self._schema_to_exp_cfg(schema, ctx)
        return FakeFreqExp().run(cfg)

    def get_analyze_params(self) -> dict[str, ParamSpec]:
        return {
            "model_type": ParamSpec(
                label="Model type",
                default="hm",
                type=str,
                choices=["hm", "t", "auto"],
            ),
            "fit_bg_slope": ParamSpec(
                label="Fit background slope",
                default=False,
                type=bool,
                choices=None,
            ),
        }

    def analyze(
        self, result: FreqRunResult, ctx: ExpContext, **user_params: Any
    ) -> FakeFreqAnalyzeResult:
        _model_type = str(user_params.get("model_type", "hm"))
        fit_bg_slope = bool(user_params.get("fit_bg_slope", False))
        model_type = cast(Literal["hm", "t", "auto"], _model_type)
        freq, fwhm, params, figure = FreqExp().analyze(
            (result.freqs, result.signals),
            model_type=model_type,
            fit_bg_slope=fit_bg_slope,
        )
        return FakeFreqAnalyzeResult(
            freq=freq,
            fwhm=fwhm,
            params=params,
            figure=figure,
            run_result=result,
        )

    def get_figure(self, analyze_result: FakeFreqAnalyzeResult) -> Optional[Figure]:
        return analyze_result.figure

    def get_writeback_items(
        self, analyze_result: FakeFreqAnalyzeResult, ctx: ExpContext
    ) -> Sequence[MetaDictWriteback | ModuleWriteback | WaveformWriteback]:
        freq = analyze_result.freq
        fwhm = analyze_result.fwhm
        md = ctx.md

        cfg = analyze_result.run_result.cfg_snapshot
        readout = cfg.modules.get("readout")
        pulse_ch = getattr(ctx.md, "res_ch", 0)
        ro_ch = getattr(ctx.md, "ro_ch", 0)

        new_readout = build_readout_for_frequency(
            readout,
            freq=freq,
            pulse_ch=pulse_ch,
            ro_ch=ro_ch,
            ml=ctx.ml,
        )
        cur_val_rf = ctx.ml.modules.get("readout_rf")

        wav_len = getattr(ctx.md, "res_probe_len", 5.0)
        new_waveform = build_waveform_for_length(
            readout,
            length=float(wav_len),
            ml=ctx.ml,
        )
        cur_val_ro = ctx.ml.waveforms.get("ro_waveform")

        return [
            MetaDictWriteback(
                key="r_f",
                description="Resonator frequency (MHz)",
                current_value=getattr(md, "r_f", None),
                md_key="r_f",
                proposed_value=round(freq, 4),
            ),
            MetaDictWriteback(
                key="rf_w",
                description="Resonator linewidth FWHM (MHz)",
                current_value=getattr(md, "rf_w", None),
                md_key="rf_w",
                proposed_value=round(fwhm, 4),
            ),
            ModuleWriteback(
                key="readout_rf",
                description="readout_rf module config",
                current_value=cur_val_rf,
                module_name="readout_rf",
                proposed_module=new_readout,
                edit_schema=make_readout_edit_template(
                    readout,
                    freq=freq,
                    pulse_ch=pulse_ch,
                    ro_ch=ro_ch,
                ),
            ),
            WaveformWriteback(
                key="ro_waveform",
                description="ro_waveform length config",
                current_value=cur_val_ro,
                waveform_name="ro_waveform",
                proposed_waveform=new_waveform,
                edit_schema=make_flat_top_waveform_edit_template(length=float(wav_len)),
            ),
        ]

    def make_save_paths(self, ctx: ExpContext) -> SavePaths:
        ts = time.strftime("%m%d")
        filename = f"{ctx.res_name}_freq_{ts}"

        if ctx.database_path:
            save_dir = create_datafolder(ctx.database_path)
            data_path = os.path.join(save_dir, filename)
        else:
            data_path = f"/tmp/{filename}"

        # image: result_dir/exps/{active_label}/image/ — mirrors em.flux_dir/image/
        if ctx.result_dir and ctx.active_label:
            flux_image_dir = os.path.join(
                ctx.result_dir, "exps", ctx.active_label, "image"
            )
            os.makedirs(flux_image_dir, exist_ok=True)
            image_path = os.path.join(flux_image_dir, f"{filename}.png")
        elif ctx.result_dir:
            image_path = os.path.join(
                ctx.result_dir, "exps", "image", f"{filename}.png"
            )
        else:
            image_path = f"/tmp/{filename}.png"

        return SavePaths(data_path=data_path, image_path=image_path)

    def save(self, data_path: str, result: FreqRunResult, ctx: ExpContext) -> None:
        pass  # no real hardware, skip HDF5 persistence

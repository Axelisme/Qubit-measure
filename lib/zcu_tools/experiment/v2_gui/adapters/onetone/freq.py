"""FakeFreqAdapter — simulates a one-tone frequency sweep using HangerModel.

FakeFreqExp mirrors the structure of FreqExp exactly (run_task + Task + LivePlot1D),
with a measure_fn that generates HangerModel signals plus Gaussian noise instead of
calling real hardware.  FakeFreqAdapter wraps FakeFreqExp and converts its flat
CfgSchema into the FakeFreqCfg that FakeFreqExp expects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Callable

from matplotlib.figure import Figure
from zcu_tools.experiment.base import AbsExperiment
from zcu_tools.experiment.v2.onetone.freq import FreqExp
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    CfgSchema,
    CfgSection,
    ExpContext,
    ModuleRefField,
    ParamSpec,
    SavePaths,
    ScalarField,
    SweepField,
    WritebackItem,
    schema_to_dict,
)
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import ProgramV2Cfg
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

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:  # noqa: ARG002
        # custom_template: nested format matching ModuleCfgFactory.from_raw() / to_dict()
        # Used when user selects <Custom> — expanded_content is deepcopy of this.
        custom_tmpl = CfgSection(
            label="Custom Readout",
            fields={
                "type": ScalarField(
                    value="readout/pulse", label="Type", type=str, editable=False
                ),
                "pulse_cfg": CfgSection(
                    label="Pulse Cfg",
                    fields={
                        "ch": ScalarField(value=0, label="Gen ch", type=int),
                        "nqz": ScalarField(value=1, label="NQZ", type=int),
                        "freq": ScalarField(
                            value=6000.0, label="Freq (MHz)", type=float
                        ),
                        "gain": ScalarField(value=0.5, label="Gain", type=float),
                    },
                ),
                "ro_cfg": CfgSection(
                    label="RO Cfg",
                    fields={
                        "ro_ch": ScalarField(value=0, label="RO ch", type=int),
                        "ro_length": ScalarField(
                            value=1.0, label="RO length (us)", type=float
                        ),
                    },
                ),
            },
        )

        from zcu_tools.gui.adapter import module_cfg_to_section
        from zcu_tools.program.v2 import AbsReadoutCfg

        available_modules = []
        if ctx.ml is not None:
            available_modules = [
                name
                for name, mod in ctx.ml.modules.items()
                if isinstance(mod, AbsReadoutCfg)
            ]

        module_name = None
        if "readout_rf" in available_modules:
            module_name = "readout_rf"
        elif available_modules:
            module_name = available_modules[0]

        expanded_content = None
        if module_name is not None and ctx.ml is not None:
            try:
                mod_cfg = ctx.ml.get_module(module_name)
                expanded_content = module_cfg_to_section(mod_cfg)
            except Exception:
                pass

        if expanded_content is None:
            import copy

            expanded_content = copy.deepcopy(custom_tmpl)

        readout_ref = ModuleRefField(
            module_name=module_name,
            override={},
            inline_cfg=None,
            expanded_content=expanded_content,
            available_modules=available_modules,
            custom_template=custom_tmpl,
            type_filter=AbsReadoutCfg,
        )

        modules_section = CfgSection(
            label="Modules", collapsible=True, fields={"readout": readout_ref}
        )
        root = CfgSection(
            fields={
                "reps": ScalarField(value=100, label="Reps", type=int),
                "rounds": ScalarField(value=100, label="Rounds", type=int),
                "freq": SweepField(
                    start=5800.0,
                    stop=6200.0,
                    expts=201,
                    label="Freq (MHz)",
                ),
                # HangerModel parameters
                "res_freq": ScalarField(
                    value=6000.0, label="Resonator freq (MHz)", type=float
                ),
                "Ql": ScalarField(value=150, label="Ql (loaded Q)", type=int),
                "Qc_abs": ScalarField(value=600, label="|Qc| (coupling Q)", type=int),
                "phi": ScalarField(value=0.0, label="phi (rad)", type=float),
                "a0_abs": ScalarField(
                    value=1.0, label="|a0| (bg amplitude)", type=float
                ),
                "edelay": ScalarField(value=0.05, label="edelay (us)", type=float),
                "noise_scale": ScalarField(value=0.05, label="Noise scale", type=float),
                "modules": modules_section,
            }
        )
        return CfgSchema(root=root)

    def _schema_to_exp_cfg(self, schema: CfgSchema, ctx: ExpContext) -> FakeFreqCfg:
        d = schema_to_dict(schema, ctx.ml)
        sweep_cfg: SweepCfg = d["freq"]  # SweepCfg pydantic object from SweepField
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
            modules=d.get("modules", {}),
            fast_mode=bool(d.get("fast_mode", False)),
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
        self,
        result: FreqRunResult,
        ctx: ExpContext,  # noqa: ARG002
        **user_params: Any,
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

    def get_writeback_spec(
        self,
        analyze_result: FakeFreqAnalyzeResult,
        ctx: ExpContext,
    ) -> list[WritebackItem]:
        freq = analyze_result.freq
        fwhm = analyze_result.fwhm
        md = ctx.md
        items = [
            WritebackItem(
                key="r_f",
                target="md",
                current_value=getattr(md, "r_f", None),
                new_value=round(freq, 4),
                description="Resonator frequency (MHz)",
            ),
            WritebackItem(
                key="rf_w",
                target="md",
                current_value=getattr(md, "rf_w", None),
                new_value=round(fwhm, 4),
                description="Resonator linewidth FWHM (MHz)",
            ),
        ]

        if ctx.ml is not None:
            from zcu_tools.gui.cfg_schemas import (
                make_flat_top_waveform_schema,
                make_pulse_readout_schema,
            )

            cfg = analyze_result.run_result.cfg_snapshot

            # 1. readout_rf module
            new_readout = _build_readout(cfg, freq, ctx)
            cur_val_rf = ctx.ml.modules.get("readout_rf")
            items.append(
                WritebackItem(
                    key="readout_rf",
                    target="ml",
                    current_value=cur_val_rf,
                    new_value=new_readout,
                    description="readout_rf module config",
                    edit_template=make_pulse_readout_schema(
                        pulse_ch=getattr(ctx.md, "res_ch", 0),
                        pulse_nqz=2,
                        pulse_freq=freq,
                        pulse_gain=0.2,
                        ro_ch=getattr(ctx.md, "ro_ch", 0),
                        ro_length=0.9,
                        trig_offset=0.335,
                    ),
                )
            )

            # 2. ro_waveform
            wav_len = getattr(ctx.md, "res_probe_len", 5.0)
            new_waveform = _build_waveform(cfg, wav_len, ctx)
            cur_val_ro = ctx.ml.waveforms.get("ro_waveform")
            items.append(
                WritebackItem(
                    key="ro_waveform",
                    target="ml",
                    current_value=cur_val_ro,
                    new_value=new_waveform,
                    description="ro_waveform length config",
                    edit_template=make_flat_top_waveform_schema(
                        length=float(wav_len),
                        raise_style="cosine",
                        raise_length=0.1,
                    ),
                )
            )
        return items

    def apply_writeback(
        self,
        ctx: ExpContext,
        analyze_result: FakeFreqAnalyzeResult,
        selected_keys: list[str],
        overrides: Optional[dict[str, Any]] = None,
    ) -> None:
        freq = analyze_result.freq
        fwhm = analyze_result.fwhm
        cfg = analyze_result.run_result.cfg_snapshot
        _overrides = overrides or {}

        if "r_f" in selected_keys:
            ctx.md.r_f = freq
        if "rf_w" in selected_keys:
            ctx.md.rf_w = fwhm

        if ctx.ml is not None:
            dirty = False
            if "readout_rf" in selected_keys:
                try:
                    raw_override = _overrides.get("readout_rf")
                    if raw_override is not None:
                        from zcu_tools.program.v2 import ModuleCfgFactory

                        new_readout = ModuleCfgFactory.from_raw(raw_override, ml=ctx.ml)
                    else:
                        new_readout = _build_readout(cfg, freq, ctx)
                    if new_readout is not None:
                        ctx.ml.register_module(readout_rf=cast(Any, new_readout))
                        dirty = True
                except Exception:
                    pass
            if "ro_waveform" in selected_keys:
                try:
                    raw_override = _overrides.get("ro_waveform")
                    if raw_override is not None:
                        from zcu_tools.program.v2 import WaveformCfgFactory

                        new_waveform = WaveformCfgFactory.from_raw(
                            raw_override, ml=ctx.ml
                        )
                    else:
                        wav_len = getattr(ctx.md, "res_probe_len", 5.0)
                        new_waveform = _build_waveform(cfg, wav_len, ctx)
                    if new_waveform is not None:
                        ctx.ml.register_waveform(ro_waveform=cast(Any, new_waveform))
                        dirty = True
                except Exception:
                    pass
            if dirty:
                try:
                    ctx.ml.dump()
                except Exception:
                    pass

    def make_save_paths(self, ctx: ExpContext) -> SavePaths:
        import os
        import time as _time

        from zcu_tools.utils.datasaver import create_datafolder

        ts = _time.strftime("%m%d")
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

    def save(
        self,
        data_path: str,  # noqa: ARG002
        result: FreqRunResult,  # noqa: ARG002
        ctx: ExpContext,  # noqa: ARG002
    ) -> None:
        pass  # no real hardware, skip HDF5 persistence


# ---------------------------------------------------------------------------
# Internal helpers — shared between get_writeback_spec and apply_writeback
# ---------------------------------------------------------------------------


def _build_readout(
    cfg: FakeFreqCfg,
    freq: float,
    ctx: ExpContext,
) -> Any:
    """Build an updated readout module with the fitted frequency."""
    gui_readout = cfg.modules.get("readout")

    if gui_readout is not None:
        try:
            pulse_cfg = getattr(gui_readout, "pulse_cfg", None)
            if pulse_cfg is not None:
                updated_pulse = pulse_cfg.with_updates(freq=freq)
                return gui_readout.with_updates(pulse_cfg=updated_pulse)
            return gui_readout
        except Exception:
            pass

    # Fallback: build from scratch
    fallback_raw = {
        "type": "readout/pulse",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": 1.0},
            "ch": getattr(ctx.md, "res_ch", 0),
            "nqz": 2,
            "freq": freq,
            "gain": 0.2,
        },
        "ro_cfg": {
            "ro_ch": getattr(ctx.md, "ro_ch", 0),
            "ro_length": 1.0,
            "trig_offset": 0.5,
        },
    }
    try:
        from zcu_tools.program.v2 import ModuleCfgFactory

        return ModuleCfgFactory.from_raw(fallback_raw, ml=ctx.ml)
    except Exception:
        return None


def _build_waveform(
    cfg: FakeFreqCfg,
    wav_len: float,
    ctx: ExpContext,
) -> Any:
    """Build an updated ro_waveform with the given probe length."""
    gui_readout = cfg.modules.get("readout")

    if gui_readout is not None:
        try:
            pulse_cfg = getattr(gui_readout, "pulse_cfg", None)
            if pulse_cfg is not None:
                gui_waveform = getattr(pulse_cfg, "waveform", None)
                if gui_waveform is not None:
                    from zcu_tools.program.v2 import AbsWaveformCfg

                    if isinstance(gui_waveform, AbsWaveformCfg):
                        return gui_waveform.with_updates(length=wav_len)
        except Exception:
            pass

    # Fallback
    fallback_wav_raw = {
        "style": "flat_top",
        "raise_waveform": {"style": "cosine", "length": 0.1},
        "length": wav_len,
    }
    try:
        from zcu_tools.program.v2 import WaveformCfgFactory

        return WaveformCfgFactory.from_raw(fallback_wav_raw, ml=ctx.ml)
    except Exception:
        return None

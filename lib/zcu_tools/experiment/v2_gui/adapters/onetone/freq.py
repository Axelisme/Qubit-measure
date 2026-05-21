"""FakeFreqAdapter — simulates a one-tone frequency sweep using HangerModel.

FakeFreqExp mirrors the structure of FreqExp exactly (run_task + Task + LivePlot1D),
with a measure_fn that generates HangerModel signals plus Gaussian noise instead of
calling real hardware.  FakeFreqAdapter wraps FakeFreqExp and converts its flat
CfgSchema into the FakeFreqCfg that FakeFreqExp expects.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, cast

logger = logging.getLogger(__name__)

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Callable

from zcu_tools.experiment.base import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.onetone.freq import FreqExp
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    ChannelValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    ModuleRefSpec,
    ModuleRefValue,
    ParamSpec,
    SavePaths,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    WritebackItem,
    make_default_value,
    schema_to_dict,
)
from zcu_tools.gui.specs.readout import DIRECT_READOUT_SPEC, PULSE_READOUT_SPEC
from zcu_tools.gui.specs.waveform import FLAT_TOP_WAVEFORM_SPEC
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

    def __init__(self, fast_mode: bool = False) -> None:
        self._fast_mode = fast_mode

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:  # noqa: ARG002
        # Spec: static structure, shared across all FakeFreqAdapter instances
        readout_spec = ModuleRefSpec(
            allowed=[DIRECT_READOUT_SPEC, PULSE_READOUT_SPEC],
            label="Readout",
        )
        modules_spec = CfgSectionSpec(
            label="Modules",
            collapsible=True,
            fields={"readout": readout_spec},
        )
        root_spec = CfgSectionSpec(
            fields={
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
                "modules": modules_spec,
            }
        )

        # Value: initial values (may come from ctx)
        # Try to pre-select readout_rf from ml if available
        chosen_key = "<Custom:Direct Readout>"
        readout_val = make_default_value(DIRECT_READOUT_SPEC)
        if ctx.ml is not None:
            try:
                from zcu_tools.gui.cfg_schemas import module_cfg_to_value
                from zcu_tools.program.v2 import AbsReadoutCfg

                # Prefer "readout_rf" by name; fall back to the last valid readout module
                candidates = {
                    name: mod
                    for name, mod in ctx.ml.modules.items()
                    if isinstance(mod, AbsReadoutCfg)
                }
                pick = candidates.get("readout_rf") or (
                    next(iter(reversed(list(candidates.values()))), None)
                )
                pick_name = (
                    "readout_rf"
                    if "readout_rf" in candidates
                    else next(iter(reversed(list(candidates.keys()))), None)
                )
                if pick is not None and pick_name is not None:
                    _, readout_val = module_cfg_to_value(pick)
                    chosen_key = pick_name
            except Exception:
                pass

        modules_val = CfgSectionValue(
            fields={
                "readout": ModuleRefValue(chosen_key=chosen_key, value=readout_val),
            }
        )
        # Pre-fill model params from md if available
        r_f: float = 6000.0
        rf_w: Optional[float] = None
        if ctx.md is not None:
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
                "modules": modules_val,
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def _schema_to_exp_cfg(self, schema: CfgSchema, ctx: ExpContext) -> FakeFreqCfg:
        d = schema_to_dict(schema, ctx.ml)
        sweep_cfg: SweepCfg = d["freq"]  # SweepCfg from make_sweep via SweepValue
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
                    edit_template=_make_pulse_readout_template(
                        pulse_ch=getattr(ctx.md, "res_ch", 0),
                        pulse_freq=freq,
                        ro_ch=getattr(ctx.md, "ro_ch", 0),
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
                    edit_template=_make_flat_top_waveform_template(
                        length=float(wav_len)
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
                    ov = _overrides.get("readout_rf")
                    if ov is not None:
                        from zcu_tools.program.v2 import ModuleCfgFactory

                        name = ov["name"] if isinstance(ov, dict) else "readout_rf"
                        raw_cfg = ov["cfg"] if isinstance(ov, dict) else ov
                        new_readout = ModuleCfgFactory.from_raw(raw_cfg, ml=ctx.ml)
                    else:
                        name = "readout_rf"
                        new_readout = _build_readout(cfg, freq, ctx)
                    if new_readout is not None:
                        ctx.ml.register_module(**{name: cast(Any, new_readout)})
                        logger.debug("apply_writeback: registered module %r", name)
                        dirty = True
                except Exception as e:
                    logger.warning("apply_writeback: readout_rf failed: %s", e)
            if "ro_waveform" in selected_keys:
                try:
                    ov = _overrides.get("ro_waveform")
                    if ov is not None:
                        from zcu_tools.program.v2 import WaveformCfgFactory

                        name = ov["name"] if isinstance(ov, dict) else "ro_waveform"
                        raw_cfg = ov["cfg"] if isinstance(ov, dict) else ov
                        new_waveform = WaveformCfgFactory.from_raw(raw_cfg, ml=ctx.ml)
                    else:
                        name = "ro_waveform"
                        wav_len = getattr(ctx.md, "res_probe_len", 5.0)
                        new_waveform = _build_waveform(cfg, wav_len, ctx)
                    if new_waveform is not None:
                        ctx.ml.register_waveform(**{name: cast(Any, new_waveform)})
                        logger.debug("apply_writeback: registered waveform %r", name)
                        dirty = True
                except Exception as e:
                    logger.warning("apply_writeback: ro_waveform failed: %s", e)
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


def _make_pulse_readout_template(
    pulse_ch: int,
    pulse_freq: float,
    ro_ch: int,
) -> CfgSchema:
    """Build a CfgSchema edit_template for a pulse readout module."""
    from zcu_tools.gui.adapter import (
        WaveformRefValue,
        make_default_value,
    )
    from zcu_tools.gui.specs.waveform import CONST_WAVEFORM_SPEC

    const_val = make_default_value(CONST_WAVEFORM_SPEC)
    const_val.fields["length"] = ScalarValue(1.0)
    pulse_val = CfgSectionValue(
        fields={
            "waveform": WaveformRefValue(
                chosen_key="<Custom:Const>",
                value=const_val,
            ),
            "ch": ChannelValue(chosen=pulse_ch, resolved=None),
            "nqz": ScalarValue(2),
            "freq": ScalarValue(pulse_freq),
            "phase": ScalarValue(0.0),
            "gain": ScalarValue(0.2),
            "pre_delay": ScalarValue(0.0),
            "post_delay": ScalarValue(0.0),
        }
    )
    ro_val = CfgSectionValue(
        fields={
            "ro_ch": ChannelValue(chosen=ro_ch, resolved=None),
            "ro_freq": ScalarValue(pulse_freq),
            "ro_length": ScalarValue(0.9),
            "trig_offset": ScalarValue(0.335),
        }
    )
    value = CfgSectionValue(
        fields={
            "pulse_cfg": pulse_val,
            "ro_cfg": ro_val,
        }
    )
    return CfgSchema(spec=PULSE_READOUT_SPEC, value=value)


def _make_flat_top_waveform_template(length: float) -> CfgSchema:
    """Build a CfgSchema edit_template for a flat_top waveform."""
    from zcu_tools.gui.adapter import WaveformRefValue
    from zcu_tools.gui.specs.waveform import COSINE_WAVEFORM_SPEC

    raise_val = CfgSectionValue(
        fields={
            "style": ScalarValue("cosine"),
            "length": ScalarValue(0.1),
        }
    )
    value = CfgSectionValue(
        fields={
            "style": ScalarValue("flat_top"),
            "length": ScalarValue(length),
            "raise_waveform": WaveformRefValue(
                chosen_key="<Custom:Cosine>",
                value=raise_val,
            ),
        }
    )
    return CfgSchema(spec=FLAT_TOP_WAVEFORM_SPEC, value=value)

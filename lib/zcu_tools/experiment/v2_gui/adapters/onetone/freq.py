"""FakeFreqAdapter — simulates a one-tone frequency sweep using HangerModel.

FakeFreqExp mirrors the structure of FreqExp exactly (run_task + Task + LivePlot1D),
with a measure_fn that generates HangerModel signals plus Gaussian noise instead of
calling real hardware.  FakeFreqAdapter wraps FakeFreqExp and converts its flat
CfgSchema into the FakeFreqCfg that FakeFreqExp expects.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Optional, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Callable

from matplotlib.figure import Figure
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.base import AbsExperiment
from zcu_tools.experiment.v2.onetone.freq import FreqExp, FreqResult
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    CfgSchema,
    CfgSection,
    ExpContext,
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

# AnalyzeResult: (freq_MHz, fwhm_MHz, param_dict, figure)
FakeFreqAnalyzeResult = tuple[float, float, dict[str, Any], Figure]


# ---------------------------------------------------------------------------
# FakeFreqCfg — same structure as FreqCfg but with HangerModel params
# ---------------------------------------------------------------------------


class FakeFreqSweepCfg(ConfigBase):
    freq: SweepCfg


class FakeFreqModelCfg(ConfigBase):
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

        self.last_cfg = cfg
        self.last_result = (freqs, signals)

        return freqs, signals

    def analyze(
        self,
        result: Optional[FreqResult] = None,
        *,
        model_type: Literal["hm", "t", "auto"] = "auto",
        fit_bg_slope: bool = False,
    ) -> tuple[float, float, dict[str, Any], Figure]:
        if result is None:
            result = self.last_result
        assert result is not None
        return FreqExp().analyze(
            result, model_type=model_type, fit_bg_slope=fit_bg_slope
        )


# ---------------------------------------------------------------------------
# FakeFreqAdapter — wraps FakeFreqExp, converts CfgSchema → FakeFreqCfg
# ---------------------------------------------------------------------------


class FakeFreqAdapter(AbsExpAdapter[FreqResult, FakeFreqAnalyzeResult]):
    """Simulated one-tone frequency sweep.  No hardware required."""

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:  # noqa: ARG002
        readout_section = CfgSection(
            label="Readout",
            fields={
                "ch": ScalarField(value=0, label="Gen ch", type=int, editable=False),
                "nqz": ScalarField(value=1, label="NQZ", type=int, editable=False),
                "freq": ScalarField(
                    value=6000.0, label="Freq (MHz)", type=float, editable=False
                ),
                "gain": ScalarField(
                    value=0.5, label="Gain", type=float, editable=False
                ),
                "ro_ch": ScalarField(value=0, label="RO ch", type=int, editable=False),
                "ro_length": ScalarField(
                    value=1.0, label="RO length (us)", type=float, editable=False
                ),
            },
        )
        modules_section = CfgSection(
            label="Modules (hardware — not used in simulation)",
            collapsible=True,
            fields={"readout": readout_section},
        )
        root = CfgSection(
            fields={
                "reps": ScalarField(value=100, label="Reps", type=int),
                "rounds": ScalarField(value=10, label="Rounds", type=int),
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
                "Ql": ScalarField(value=5000, label="Ql (loaded Q)", type=int),
                "Qc_abs": ScalarField(value=6000, label="|Qc| (coupling Q)", type=int),
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
        )

    def get_run_params(self) -> dict[str, ParamSpec]:
        return {}

    def run(
        self,
        ctx: ExpContext,
        schema: CfgSchema,
        **user_params: Any,  # noqa: ARG002
    ) -> FreqResult:
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
        result: FreqResult,
        ctx: ExpContext,  # noqa: ARG002
        **user_params: Any,
    ) -> FakeFreqAnalyzeResult:
        _model_type = str(user_params.get("model_type", "hm"))
        fit_bg_slope = bool(user_params.get("fit_bg_slope", False))
        model_type = cast(Literal["hm", "t", "auto"], _model_type)
        return FreqExp().analyze(
            result, model_type=model_type, fit_bg_slope=fit_bg_slope
        )

    def get_figure(self, analyze_result: FakeFreqAnalyzeResult) -> Optional[Figure]:
        return analyze_result[3]

    def get_writeback_spec(
        self,
        analyze_result: FakeFreqAnalyzeResult,
        ctx: ExpContext,
    ) -> list[WritebackItem]:
        freq, fwhm, _, _ = analyze_result
        md = ctx.md
        return [
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

    def apply_writeback(
        self,
        ctx: ExpContext,
        analyze_result: FakeFreqAnalyzeResult,
        selected_keys: list[str],
    ) -> None:
        freq, fwhm, _, _ = analyze_result
        if "r_f" in selected_keys:
            ctx.md.r_f = freq
        if "rf_w" in selected_keys:
            ctx.md.rf_w = fwhm

    def make_save_paths(self, ctx: ExpContext) -> SavePaths:  # noqa: ARG002
        import time as _time

        ts = _time.strftime("%m%d")
        return SavePaths(
            data_path=f"/tmp/fake_freq_{ts}",
            image_path=f"/tmp/fake_freq_{ts}.png",
        )

    def save(
        self,
        data_path: str,  # noqa: ARG002
        result: FreqResult,  # noqa: ARG002
        ctx: ExpContext,  # noqa: ARG002
    ) -> None:
        pass  # no real hardware, skip HDF5 persistence

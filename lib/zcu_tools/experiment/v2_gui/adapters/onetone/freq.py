"""FakeFreqAdapter — simulates a one-tone frequency sweep using HangerModel.

The run() generates ideal HangerModel signals plus complex Gaussian noise
whose amplitude scales as 1/sqrt(reps * rounds), matching the SNR improvement
of real hardware averaging.  analyze() delegates to FreqExp.analyze(), so
the fitting pipeline is identical to the real experiment.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Optional, cast

import numpy as np
from matplotlib.figure import Figure
from zcu_tools.experiment.v2.onetone.freq import FreqExp, FreqResult
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    CfgSchema,
    CfgSection,
    ExpContext,
    ParamSpec,
    SavePaths,
    ScalarField,
    WritebackItem,
    schema_to_dict,
)
from zcu_tools.utils.fitting.resonance.hanger import HangerModel

# AnalyzeResult: (freq_MHz, fwhm_MHz, param_dict, figure)
FakeFreqAnalyzeResult = tuple[float, float, dict[str, Any], Figure]


class FakeFreqAdapter(AbsExpAdapter[FreqResult, FakeFreqAnalyzeResult]):
    """Simulated one-tone frequency sweep.  No hardware required."""

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:  # noqa: ARG002
        root = CfgSection(
            fields={
                "reps": ScalarField(value=100, label="Reps", type=int),
                "rounds": ScalarField(value=10, label="Rounds", type=int),
                # Frequency sweep range
                "freq_start": ScalarField(
                    value=5.8, label="Freq start (MHz)", type=float
                ),
                "freq_stop": ScalarField(
                    value=6.2, label="Freq stop (MHz)", type=float
                ),
                "freq_expts": ScalarField(value=201, label="Freq points", type=int),
                # Hanger model parameters
                "freq": ScalarField(
                    value=6.0, label="Resonator freq (MHz)", type=float
                ),
                "Ql": ScalarField(value=5000, label="Ql (loaded Q)", type=int),
                "Qc_abs": ScalarField(value=6000, label="|Qc| (coupling Q)", type=int),
                "phi": ScalarField(value=0.0, label="phi (rad)", type=float),
                "a0_abs": ScalarField(
                    value=1.0, label="|a0| (bg amplitude)", type=float
                ),
                "edelay": ScalarField(value=0.05, label="edelay (us)", type=float),
                "noise_scale": ScalarField(value=0.05, label="Noise scale", type=float),
            }
        )
        return CfgSchema(root=root)

    def get_run_params(self) -> dict[str, ParamSpec]:
        return {}

    def run(
        self,
        ctx: ExpContext,  # noqa: ARG002
        schema: CfgSchema,
        **user_params: Any,  # noqa: ARG002
    ) -> FreqResult:
        d = schema_to_dict(schema, ctx.ml)

        reps = int(d.get("reps", 100))
        rounds = int(d.get("rounds", 10))
        freq_start = float(d.get("freq_start", 5.8))
        freq_stop = float(d.get("freq_stop", 6.2))
        freq_expts = int(d.get("freq_expts", 201))
        freq = float(d.get("freq", 6.0))
        Ql = float(d.get("Ql", 5000))
        Qc_abs = float(d.get("Qc_abs", 6000))
        phi = float(d.get("phi", 0.0))
        a0_abs = float(d.get("a0_abs", 1.0))
        edelay = float(d.get("edelay", 0.05))
        noise_scale = float(d.get("noise_scale", 0.05))

        freqs = np.linspace(freq_start, freq_stop, freq_expts)
        a0 = complex(a0_abs)
        Qc = complex(Qc_abs * np.exp(-1j * phi))

        # ideal signal from HangerModel (Qc is complex despite the float hint)
        clean = HangerModel.calc_signals(
            freqs, freq, Ql, cast(float, Qc), phi, a0, edelay
        )

        # noise amplitude decreases with sqrt(reps * rounds) — same as hardware averaging
        sigma = noise_scale / np.sqrt(reps * rounds)
        rng = np.random.default_rng()
        noise = rng.normal(0, sigma, len(freqs)) + 1j * rng.normal(0, sigma, len(freqs))

        # simulate round-level accumulation: brief sleep so the run feels non-instant
        time.sleep(0.05)

        signals = (clean + noise).astype(np.complex128)
        return freqs, signals

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
        exp = FreqExp()
        return exp.analyze(result, model_type=model_type, fit_bg_slope=fit_bg_slope)

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

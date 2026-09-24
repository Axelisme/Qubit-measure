from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.singleshot import AcStarkCfg, AcStarkExp
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    custom,
    md_get_float,
    md_has_key,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CfgSchema,
    DirectValue,
    EvalValue,
    SweepValue,
)
from zcu_tools.program.v2.modules.pulse import PulseCfg

from ._shared import read_chi_kappa, read_ge_centers, readout_probe_freq

# Domain AcStarkExp.analyze returns (ac_coeff, fig). The fitted AC-Stark
# coefficient is written back to the MetaDict (key ``ac_stark_coeff``, matching
# single_qubit.md:3329) — it is the photon-number-per-gain² calibration the
# downstream MIST experiments read as ``ac_coeff``.
SsAcStarkRunResult: TypeAlias = Any  # AcStarkResult (frozen domain dataclass)

_RF_WIDTH_FALLBACK_MHZ = 5.0


def _selected_qubit_pulse(ctx: ExpContext) -> PulseCfg | None:
    # Match the pi_pulse role's library priority for stark_pulse2.
    for name in ("pi_amp", "pi_len"):
        pulse = ctx.ml.modules.get(name)
        if isinstance(pulse, PulseCfg):
            return pulse
    return None


def _probe_length(ctx: ExpContext) -> float:
    pulse = _selected_qubit_pulse(ctx)
    if pulse is not None:
        length = pulse.waveform.length
        if (
            not isinstance(length, (int, float))
            or not math.isfinite(length)
            or length <= 0
        ):
            raise ValueError(
                "The selected pi-pulse waveform length must be positive and finite"
            )
        return float(length)
    return 0.3  # Draft fallback until a calibrated pi pulse is available.


def _qubit_frequency(ctx: ExpContext) -> float | EvalValue:
    if md_has_key(ctx, "q_f"):
        return EvalValue(expr="q_f")
    pulse = _selected_qubit_pulse(ctx)
    if pulse is None:
        return 4000.0
    if not isinstance(pulse.freq, (int, float)):
        raise ValueError("The selected pi-pulse frequency must be a fixed number")
    return float(pulse.freq)


def _qubit_channel(ctx: ExpContext) -> int | EvalValue:
    for key in ("qub_ch", "qub_1_4_ch", "qub_4_5_ch"):
        if md_has_key(ctx, key):
            return EvalValue(expr=key)
    pulse = _selected_qubit_pulse(ctx)
    return pulse.ch if pulse is not None else 0


def _qubit_mixer_frequency(ctx: ExpContext) -> DirectValue | EvalValue:
    pulse = _selected_qubit_pulse(ctx)
    mixer_freq = pulse.mixer_freq if pulse is not None else None
    if mixer_freq is None:
        return DirectValue(None)
    if md_has_key(ctx, "q_f"):
        return EvalValue(expr="q_f")
    return DirectValue(mixer_freq)


def _rf_width_timing(
    ctx: ExpContext, *, numerator: float, offset: float = 0.0
) -> float | EvalValue:
    coefficient = numerator / (2 * math.pi)
    if md_has_key(ctx, "rf_w"):
        width = md_get_float(ctx, "rf_w", float("nan"))
        if not math.isfinite(width) or width <= 0:
            raise ValueError("MetaDict 'rf_w' must be a positive finite number")
        return EvalValue(expr=f"{offset} + {coefficient} / rf_w")
    return offset + coefficient / _RF_WIDTH_FALLBACK_MHZ


def _cavity_tone_length(ctx: ExpContext) -> float | EvalValue:
    return _rf_width_timing(ctx, numerator=5.1, offset=_probe_length(ctx))


def _qubit_pre_delay(ctx: ExpContext) -> float | EvalValue:
    return _rf_width_timing(ctx, numerator=5.0)


def _qubit_post_delay(ctx: ExpContext) -> float | EvalValue:
    return _rf_width_timing(ctx, numerator=3.1)


def _freq_sweep_default(ctx: ExpContext) -> SweepValue:
    if md_has_key(ctx, "q_f"):
        return SweepValue(
            start=EvalValue(expr="q_f - 700.0"),
            stop=EvalValue(expr="q_f + 100.0"),
            expts=801,
        )
    return SweepValue(start=3300.0, stop=4100.0, expts=801)


@dataclass
class SsAcStarkAnalyzeResult(AnalyzeResultBase):
    ac_stark_coeff: float
    figure: Figure


class SsAcStarkAdapter(
    BaseAdapter[
        AcStarkCfg,
        SsAcStarkRunResult,
        SsAcStarkAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    exp_cls = AcStarkExp
    ExpCfg_cls: ClassVar[Any] = AcStarkCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot AC-Stark calibration: drives a cavity Stark tone and a "
            "qubit probe tone while sweeping cavity gain and qubit frequency (2D), "
            "classifying each shot in-program against the |g>/|e> IQ-cluster "
            "centres. Fits the Stark-shifted resonance versus gain² to extract "
            "the AC-Stark coefficient (photon number per gain²). Runs on real "
            "hardware."
        ),
        expects_md=(
            "REQUIRES the single-shot discrimination calibration in the "
            "MetaDict — run 'singleshot/ge' first and apply its writeback so "
            "'g_center' / 'e_center' / 'ge_radius' are present; the run "
            "classifies each shot against them and fast-fails if any is missing. "
            "ANALYSIS additionally REQUIRES 'chi' (dispersive shift, MHz) and "
            "'rf_w' (resonator linewidth kappa, MHz) — both feed the AC-Stark "
            "coefficient fit and analyze fast-fails if either is missing (run "
            "the dispersive-shift experiment first). 'rf_w' also sets the pulse "
            "timing; 'readout_f' (or 'r_f') seeds the cavity tone, and 'q_f' "
            "centres the qubit-frequency sweep. Qubit-channel metadata remains "
            "linked in the pulse prefill; a configured qubit mixer tracks 'q_f'. "
            "Optionally reads "
            "'confusion_matrix' (readout correction) and 'cutoff' (drop gains "
            "above it before fitting) at analyze time."
        ),
        expects_ml=(
            "Uses the calibrated 'pi_amp' qubit pulse when available (falling back "
            "to 'pi_len') and a readout module. The cavity tone is inline; reset "
            "and init pulse start disabled, as in single_qubit.md."
        ),
        typical_writeback=(
            "Proposes the fitted AC-Stark coefficient into MetaDict "
            "'ac_stark_coeff' (photon number per gain²); the MIST experiments "
            "read it to rescale their x-axis to photon number."
        ),
        recommended=(
            "Run after 'singleshot/ge' and after the dispersive-shift "
            "experiment has set 'chi' / 'rf_w'. Sweep the Stark gain across the "
            "onset and the probe frequency around the qubit line."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True, init=ModuleInit.DISABLED)
            .pulse(
                "init_pulse",
                role_id="pi_pulse",
                optional=True,
                init=ModuleInit.DISABLED,
            )
            .pulse(
                "stark_pulse1",
                role_id="res_probe",
                label="Cavity Stark Tone (gain sweep)",
                init=ModuleInit.INLINE,
                overrides={
                    "freq": custom(
                        readout_probe_freq,
                        description="cavity Stark tone frequency",
                    ),
                    "gain": 0.0,
                    "waveform.length": custom(
                        _cavity_tone_length,
                        description="cavity Stark tone length",
                    ),
                },
            )
            .pulse(
                "stark_pulse2",
                role_id="pi_pulse",
                label="Qubit Probe Tone (freq sweep)",
                blank_overrides={"waveform.length": 0.3},
                overrides={
                    "freq": custom(
                        _qubit_frequency,
                        description="qubit probe frequency from q_f",
                    ),
                    "ch": custom(
                        _qubit_channel,
                        description="qubit probe channel from MetaDict",
                    ),
                    "mixer_freq": custom(
                        _qubit_mixer_frequency,
                        description="qubit probe mixer frequency from q_f",
                    ),
                    "pre_delay": custom(
                        _qubit_pre_delay,
                        description="qubit probe pre-delay",
                    ),
                    "post_delay": custom(
                        _qubit_post_delay,
                        description="qubit probe post-delay",
                    ),
                },
            )
            .readout()
            .relax_delay(5.5)
            .sweep(
                "gain",
                label="Cavity Stark gain (a.u.)",
                default=SweepValue(start=0.0, stop=0.22, expts=301),
            )
            .sweep(
                "freq",
                label="Qubit probe frequency (MHz)",
                default=custom(
                    _freq_sweep_default,
                    description="AC-Stark probe frequency range",
                ),
            )
            .reps(1000)
            .rounds(2)
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> SsAcStarkRunResult:
        # Override the standard run path: the domain run needs the GE
        # classification trio (not in cfg) — read it from md and forward it.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        g_center, e_center, radius = read_ge_centers(req.md)
        return AcStarkExp().run(soc, soccfg, cfg, g_center, e_center, radius)

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def analyze(
        self, req: AnalyzeRequest[SsAcStarkRunResult, NoAnalyzeParams]
    ) -> SsAcStarkAnalyzeResult:
        # ``chi`` / ``kappa`` (= md 'rf_w', the resonator linewidth) are required
        # fit inputs read from md — fast-fail if either is missing. The domain
        # derives eta = kappa²/(kappa²+chi²) internally. ``confusion_matrix``
        # (readout correction) and ``cutoff`` (drop high gains before fitting) are
        # optional md inputs, never user knobs; absent → domain defaults.
        chi, kappa = read_chi_kappa(req.md)
        confusion = req.md.get("confusion_matrix")
        cutoff = req.md.get("cutoff")
        ac_coeff, fig = AcStarkExp().analyze(
            chi,
            req.run_result,
            kappa=kappa,
            confusion_matrix=confusion,
            cutoff=cutoff,
        )
        return SsAcStarkAnalyzeResult(ac_stark_coeff=ac_coeff, figure=fig)

    def get_writeback_items(
        self,
        req: WritebackRequest[SsAcStarkRunResult, SsAcStarkAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        # Key ``ac_stark_coeff`` per single_qubit.md:3329.
        return [
            MetaDictWriteback(
                target_name="ac_stark_coeff",
                description="AC-Stark coefficient (photon number per gain²)",
                proposed_value=req.analyze_result.ac_stark_coeff,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_sh_ac_stark_{time.strftime('%m%d')}"

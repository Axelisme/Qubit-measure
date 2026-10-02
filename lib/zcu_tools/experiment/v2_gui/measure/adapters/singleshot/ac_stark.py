from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.singleshot.ac_stark import (
    AcStarkAnalyzeOptions,
    AcStarkCfg,
    AcStarkExp,
    AcStarkResult,
)
from zcu_tools.experiment.v2_gui.measure.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    custom,
    md_get_float,
    md_has_key,
)
from zcu_tools.experiment.v2_gui.measure.adapters.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.cfg import (
    DirectValue,
    EvalValue,
    ScalarSpec,
    SweepValue,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2.modules.pulse import PulseCfg

from ._shared import read_chi_kappa, readout_probe_freq

# Domain analysis returns a numeric coefficient and publishes its named fit. The AC-Stark
# coefficient is written back to the MetaDict (key ``ac_stark_coeff``, matching
# single_qubit.md:3329) — it is the photon-number-per-gain² calibration the
# downstream MIST experiments read as ``ac_coeff``.
SsAcStarkRunResult: TypeAlias = RunRecord[AcStarkCfg, AcStarkResult]

_RF_WIDTH_FALLBACK_MHZ = 5.0


def _selected_qubit_pulse(ctx: SessionEnv) -> PulseCfg | None:
    # Match the pi_pulse role's library priority for stark_pulse2.
    for name in ("pi_amp", "pi_len"):
        pulse = ctx.ml.modules.get(name)
        if isinstance(pulse, PulseCfg):
            return pulse
    return None


def _probe_length(ctx: SessionEnv) -> float:
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


def _qubit_frequency(ctx: SessionEnv) -> float | EvalValue:
    if md_has_key(ctx, "q_f"):
        return EvalValue(expr="q_f")
    pulse = _selected_qubit_pulse(ctx)
    if pulse is None:
        return 4000.0
    if not isinstance(pulse.freq, (int, float)):
        raise ValueError("The selected pi-pulse frequency must be a fixed number")
    return float(pulse.freq)


def _qubit_channel(ctx: SessionEnv) -> int | EvalValue:
    for key in ("qub_ch", "qub_1_4_ch", "qub_4_5_ch"):
        if md_has_key(ctx, key):
            return EvalValue(expr=key)
    pulse = _selected_qubit_pulse(ctx)
    return pulse.ch if pulse is not None else 0


def _qubit_mixer_frequency(ctx: SessionEnv) -> DirectValue | EvalValue:
    pulse = _selected_qubit_pulse(ctx)
    mixer_freq = pulse.mixer_freq if pulse is not None else None
    if mixer_freq is None:
        return DirectValue(None)
    if md_has_key(ctx, "q_f"):
        return EvalValue(expr="q_f")
    return DirectValue(mixer_freq)


def _rf_width_timing(
    ctx: SessionEnv, *, numerator: float, offset: float = 0.0
) -> float | EvalValue:
    coefficient = numerator / (2 * math.pi)
    if md_has_key(ctx, "rf_w"):
        width = md_get_float(ctx, "rf_w", float("nan"))
        if not math.isfinite(width) or width <= 0:
            raise ValueError("MetaDict 'rf_w' must be a positive finite number")
        return EvalValue(expr=f"{offset} + {coefficient} / rf_w")
    return offset + coefficient / _RF_WIDTH_FALLBACK_MHZ


def _cavity_tone_length(ctx: SessionEnv) -> float | EvalValue:
    return _rf_width_timing(ctx, numerator=5.1, offset=_probe_length(ctx))


def _qubit_pre_delay(ctx: SessionEnv) -> float | EvalValue:
    return _rf_width_timing(ctx, numerator=5.0)


def _qubit_post_delay(ctx: SessionEnv) -> float | EvalValue:
    return _rf_width_timing(ctx, numerator=3.1)


def _freq_sweep_default(ctx: SessionEnv) -> SweepValue:
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
            "Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved "
            "cfg, not live MetaDict. Enter direct cfg values or optionally seed "
            "defaults with 'singleshot/ge' writeback. The run classifies each "
            "shot using these values; missing or invalid cfg calibration fails "
            "before hardware. "
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
            "Set calibration cfg directly or seed it with 'singleshot/ge'; run "
            "after the dispersive-shift "
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
            .field(
                "g_center",
                spec=ScalarSpec("Ground center", complex),
                default=EvalValue("g_center"),
            )
            .field(
                "e_center",
                spec=ScalarSpec("Excited center", complex),
                default=EvalValue("e_center"),
            )
            .field(
                "radius",
                spec=ScalarSpec("Classification radius", float),
                default=EvalValue("ge_radius"),
            )
            .reps(1000)
            .rounds(2)
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> SsAcStarkRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = AcStarkExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[SsAcStarkRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> SsAcStarkAnalyzeResult:
        # ``chi`` / ``kappa`` (= md 'rf_w', the resonator linewidth) are required
        # fit inputs read from md — fast-fail if either is missing. The domain
        # derives eta = kappa²/(kappa²+chi²) internally. ``confusion_matrix``
        # (readout correction) and ``cutoff`` (drop high gains before fitting) are
        # optional md inputs, never user knobs; absent → domain defaults.
        chi, kappa = read_chi_kappa(req.md)
        confusion = req.md.get("confusion_matrix")
        cutoff = req.md.get("cutoff")
        analysis = AcStarkExp().analyze(
            req.run_result,
            AcStarkAnalyzeOptions(
                chi=chi, kappa=kappa, confusion_matrix=confusion, cutoff=cutoff
            ),
            plots=plots,
        )
        return SsAcStarkAnalyzeResult(ac_stark_coeff=analysis.ac_stark_coeff)

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

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sh_ac_stark_{time.strftime('%m%d')}"

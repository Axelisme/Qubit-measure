from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import (
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2._support.measure.detune import (
    detune_ratio_of,
    resolve_detune,
    strip_detune_ratio,
)
from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import (
    SweepDefault,
    custom,
    module_seed,
    path_seed,
    scaled_path_seed,
)
from zcu_lab.v2._support.measure.spec_helpers import (
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
)
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyAnalyzeOptions,
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyResult,
)

logger = logging.getLogger(__name__)


T2RamseyRunResult: TypeAlias = RunRecord[T2RamseyCfg, T2RamseyResult]


def _blank_pulse(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(make_pulse_module_spec().allowed[0])
    value.with_field("freq", path_seed("$qubit.freq", fallback=4000.0).resolve(ctx))
    value.with_field("ch", path_seed("$qubit.wiring.L01.ch", fallback=0).resolve(ctx))
    value.with_field("nqz", 2)
    value.with_field("gain", 0.1)
    value.with_field("waveform.length", 5.1)
    return ReferenceValue(make_custom_reference_key("Pulse"), value)


def _blank_readout(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(make_pulse_readout_module_spec().allowed[0])
    frequency = path_seed("$resonator.freq", fallback=6000.0).resolve(ctx)
    value.with_field("pulse_cfg.freq", frequency)
    value.with_field(
        "pulse_cfg.ch", path_seed("$resonator.wiring.ch", fallback=0).resolve(ctx)
    )
    value.with_field("pulse_cfg.nqz", 2)
    value.with_field("pulse_cfg.gain", 0.1)
    value.with_field("pulse_cfg.waveform.length", 1.0)
    value.with_field("ro_cfg.ro_freq", frequency)
    value.with_field(
        "ro_cfg.ro_ch", path_seed("$resonator.wiring.ro_ch", fallback=0).resolve(ctx)
    )
    value.with_field("ro_cfg.ro_length", 0.9)
    value.with_field(
        "ro_cfg.trig_offset",
        path_seed(
            "$resonator.ext.time_of_fly",
            expr="$resonator.ext.time_of_fly + 0.05",
            fallback=0.55,
        ).resolve(ctx),
    )
    return ReferenceValue(make_custom_reference_key("Pulse Readout"), value)


@dataclass
class T2RamseyAnalyzeParams:
    fit_fringe: Annotated[bool, ParamMeta(label="Fit fringe")] = True
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset (fringe only)")] = (
        False
    )


@dataclass
class T2RamseyAnalyzeResult(AnalyzeResultBase):
    t2r: float
    t2r_err: float
    # ``detune`` is the fitted fringe frequency (MHz); meaningful only when the
    # fringe fit ran (``fit_fringe`` True). Decay-only fits report 0.0.
    detune: float
    fit_fringe: bool
    fit_quality: dict[str, dict[str, object]] | None = None


class T2RamseyAdapter(
    BaseAdapter[
        T2RamseyCfg,
        T2RamseyRunResult,
        T2RamseyAnalyzeResult,
        T2RamseyAnalyzeParams,
    ]
):
    exp_cls = T2RamseyExp
    ExpCfg_cls: ClassVar[Any] = T2RamseyCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "T2 Ramsey: two pi/2 pulses separated by a swept free-evolution "
            "delay; fits the decaying fringe to extract the Ramsey coherence "
            "time T2* and the qubit-drive detuning. Runs on real hardware. Run "
            "after a pi/2 pulse is calibrated; a small deliberate detuning "
            "produces the fringe whose frequency reads out the true qubit "
            "frequency offset."
        ),
        expects_md=(
            "Defaults use the tab's fixed qubit and referenced resonator. "
            "Delay ends at 1.5 * qubit.t2r (us, fallback 0.4 us); relax_delay "
            "is 5 * qubit.t1 (fallback 100 us). Inline pi/2 reads qubit.freq "
            "and qubit.wiring.L01.ch. Inline readout reads resonator.freq, "
            "resonator.wiring.ch/ro_ch, and resonator.ext.time_of_fly + "
            "0.05 us (fallback 0.55 us). Missing or null calibration uses "
            "fallback; unbound roles remain Invalid until supplied."
        ),
        expects_ml=(
            "Uses captured qubit.module.x90 and qubit.module.readout. Missing "
            "or dangling slots seed fresh inline pulse/readout defaults. "
            "Optional qubit.module.reset stays disabled when absent. Verify "
            "the inline pulse values and readout timing before Run."
        ),
        typical_writeback=(
            "Proposes fitted qubit.t2r (us) to the captured qubit and working "
            "point. A fringe fit with source cfg and known realized detune "
            "also proposes qubit.freq (MHz): source pi/2 frequency + realized "
            "detune - fitted fringe detune. A pure decay fit or a loaded "
            "source without realized detune proposes only qubit.t2r. "
            "No module writeback; later focus changes do not retarget it."
        ),
        recommended=(
            "Enable Fit phase offset to fit a free fringe phase; it defaults to "
            "fixed phase and has no effect for decay-only analysis. "
            "Set the 'Detune ratio (fringes/step)' cfg knob to a small "
            "deliberate offset (default 0.05) — it is the number of fringe "
            "periods per delay-sweep step; the absolute detune (MHz) applied to "
            "the second pi/2 pulse phase is detune_ratio / sweep step, so the "
            "Ramsey signal oscillates at that detuning. The analysis 'Fit "
            "fringe' option (on by default, matching the nonzero default ratio) "
            "fits the fringe frequency to read back the true qubit-frequency "
            "offset and write qubit.freq. Set the ratio to 0 and turn 'Fit fringe' "
            "off for a pure decay fit when driving on resonance. A delay sweep "
            "reaching a few T2* captures enough fringe periods and decay."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        """Declare shape and fresh defaults resolved on the supplied capture.

        Instantiate with SessionEnv.parameters. Bound missing/null calibration
        uses the documented fallback; unbound roles remain cfg Invalid. Source
        or module materialization faults propagate without reading live md/ml.
        """
        return (
            MeasureCfgBuilder()
            .field(
                "modules.reset",
                spec=make_reset_module_spec(optional=True),
                default=module_seed("$qubit.module.reset", fallback=None),
            )
            .field(
                "modules.pi2_pulse",
                spec=make_pulse_module_spec(),
                default=module_seed(
                    "$qubit.module.x90",
                    fallback=custom(_blank_pulse, description="inline pi/2 pulse"),
                ),
            )
            .field(
                "modules.readout",
                spec=make_readout_module_spec(),
                default=module_seed(
                    "$qubit.module.readout",
                    fallback=custom(_blank_readout, description="inline readout"),
                ),
            )
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=100.0),
            )
            .sweep(
                "length",
                label="Delay (us)",
                default=SweepDefault(
                    start=0.0,
                    stop=scaled_path_seed("$qubit.t2r", factor=1.5, fallback_value=0.4),
                    expts=101,
                ),
            )
            .float(
                "detune_ratio",
                label="Detune ratio (fringes/step)",
                default=0.05,
                decimals=3,
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> T2RamseyCfg:
        cfg = super().build_exp_cfg(strip_detune_ratio(raw_cfg), req)
        cfg.detune = resolve_detune(detune_ratio_of(raw_cfg), cfg.sweep.length.step)
        return cfg

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> T2RamseyRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = self.exp_cls().run(cfg, context=context)
        logger.info("T2Ramsey true detune: %s MHz", result.true_activate_detune)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[T2RamseyRunResult, T2RamseyAnalyzeParams],
        *,
        plots: Plots,
    ) -> T2RamseyAnalyzeResult:
        params = req.analyze_params
        analysis = T2RamseyExp().analyze(
            req.run_result,
            T2RamseyAnalyzeOptions(
                fit_fringe=params.fit_fringe, fit_phase=params.fit_phase
            ),
            plots=plots,
        )
        return T2RamseyAnalyzeResult(
            t2r=analysis.t2r,
            t2r_err=analysis.t2r_err,
            fit_quality=fit_quality_summary(analysis.fit_quality),
            detune=analysis.detune,
            fit_fringe=params.fit_fringe,
        )

    def get_writeback_items(
        self, req: WritebackRequest[T2RamseyRunResult, T2RamseyAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose captured qubit.t2r, plus qubit.freq for a sourced fringe fit.

        req supplies the committed fit, source record and captured context.
        Frequency is source pi/2 frequency + realized detune - fitted detune;
        it is omitted for decay fits, cfg=None or unknown realized detune.
        Non-scalar source frequency raises ValueError. WritebackService resolves
        destinations on req.ctx.parameters and rejects missing captures/unbound
        roles. No live calibration lookup or write occurs here.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.t2r",
                description="T2 Ramsey time (us)",
                proposed_value=result.t2r,
            ),
        ]

        # Frequency correction uses only the source run, never current calibration.
        snapshot = req.run_result.cfg
        if (
            result.fit_fringe
            and snapshot is not None
            and req.run_result.result.true_activate_detune is not None
        ):
            source_freq = snapshot.modules.pi2_pulse.freq
            if not isinstance(source_freq, float):
                raise ValueError(
                    "qubit.freq writeback requires a scalar source pi/2 frequency"
                )
            q_f = (
                source_freq + req.run_result.result.true_activate_detune - result.detune
            )
            items.append(
                ParameterWriteback(
                    target_name="$qubit.freq",
                    description="Qubit frequency (MHz)",
                    proposed_value=q_f,
                )
            )

        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_t2ramsey_{time.strftime('%m%d')}"

"""Single-shot T1-with-tone-sweep adapters (split by outer sweep axis).

The domain ``T1WithToneSweepExp`` is a 2D experiment: it always sweeps the
probe ``length`` plus exactly one *outer* axis (gain XOR freq), enforced by its
``_resolve_outer_sweep`` (it raises unless exactly one of gain/freq is set).
``T1WithToneSweepSweepCfg`` makes both gain and freq ``Optional``.

Rather than expose both optional sweeps in one form (which would let a user set
zero or two and hit the domain's run-time error — least-surprise violation),
this module splits into two adapters that each expose ``length`` + exactly one
outer sweep:

- ``SsT1ToneSweepGainAdapter`` → length + gain (registry 'singleshot/t1_tone_sweep_gain')
- ``SsT1ToneSweepFreqAdapter`` → length + freq (registry 'singleshot/t1_tone_sweep_freq')

Each lowers a cfg with only its one outer sweep set, so the domain's
``_resolve_outer_sweep`` is satisfied by construction. The shared run / analyze /
writeback / guide logic lives in ``_SsT1ToneSweepBase``; the subclasses differ
only in which outer sweep they declare + its default.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import (
    EvalValue,
    ScalarSpec,
    SweepValue,
    resolved_direct_number,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_lab.v2._support.measure.seeds import SweepDefault, custom, scaled_md
from zcu_lab.v2._support.measure.singleshot_shared import (
    readout_probe_freq,
    readout_probe_freq_range,
)
from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep.core import (
    T1WithToneSweepAnalyzeOptions,
    T1WithToneSweepCfg,
    T1WithToneSweepExp,
    T1WithToneSweepResult,
)

SsT1ToneSweepRunResult: TypeAlias = RunRecord[T1WithToneSweepCfg, T1WithToneSweepResult]


@dataclass
class SsT1ToneSweepAnalyzeResult(AnalyzeResultBase):
    pass


class _SsT1ToneSweepBase(
    BaseAdapter[
        T1WithToneSweepCfg,
        SsT1ToneSweepRunResult,
        SsT1ToneSweepAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    """Shared body for the two T1-with-tone-sweep adapters.

    Subclasses declare the single outer sweep axis (gain XOR freq) so the lowered
    cfg sets only one of them and the domain ``_resolve_outer_sweep`` is satisfied.
    """

    exp_cls = T1WithToneSweepExp
    ExpCfg_cls: ClassVar[Any] = T1WithToneSweepCfg

    # Subclass contract: which outer sweep to expose, its label/default, the probe
    # role to default it to, and the filename token.
    outer_key: ClassVar[str]
    outer_label: ClassVar[str]
    outer_default: ClassVar[SweepValue]
    filename_token: ClassVar[str]

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        outer_key = cls.outer_key
        outer_label = cls.outer_label
        outer_default = deepcopy(cls.outer_default)
        outer_expts = resolved_direct_number(outer_default.expts)
        if not isinstance(outer_expts, int):
            raise TypeError("Outer sweep default requires an integer point count")
        outer_sweep = (
            custom(
                lambda ctx: readout_probe_freq_range(ctx, outer_expts),
                description="readout probe frequency range",
            )
            if outer_key == "freq"
            else outer_default
        )
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("pi_pulse", role_id="pi_pulse")
            .pulse(
                "probe_pulse",
                role_id="res_probe",
                label="Probe Pulse",
                init=ModuleInit.INLINE,
                overrides={
                    "freq": custom(
                        readout_probe_freq,
                        description="readout probe frequency",
                    )
                },
            )
            .readout()
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.0))
            .sweep(
                "length",
                label="Delay (us)",
                default=SweepDefault(
                    start=0.0,
                    # The fallback is the fully scaled 5 * 100 us window.
                    stop=scaled_md("t1", factor=5.0, fallback_value=500.0),
                    expts=51,
                ),
            )
            .sweep(outer_key, label=outer_label, default=outer_sweep)
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
            .bool("uniform", label="Uniform (linear) sweep", default=True)
            .reps(1000)
            .rounds(10)
            .build()
        )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> SsT1ToneSweepRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, T1WithToneSweepExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[SsT1ToneSweepRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> SsT1ToneSweepAnalyzeResult:
        # ``ac_coeff`` (= md 'ac_stark_coeff') rescales the outer x-axis to photon
        # number; ``confusion_matrix`` readout-corrects; both are md inputs, not
        # user knobs. ``xlabel`` labels the rate plots with this adapter's outer
        # axis.
        ac_coeff = req.md.get("ac_stark_coeff")
        confusion = req.md.get("confusion_matrix")
        T1WithToneSweepExp().analyze(
            req.run_result,
            T1WithToneSweepAnalyzeOptions(
                ac_coeff=ac_coeff, confusion_matrix=confusion, xlabel=self.outer_label
            ),
            plots=plots,
        )
        return SsT1ToneSweepAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_t1_tone_sweep_{self.filename_token}_{time.strftime('%m%d')}"


class SsT1ToneSweepGainAdapter(_SsT1ToneSweepBase):
    outer_key: ClassVar[str] = "gain"
    outer_label: ClassVar[str] = "Probe gain (a.u.)"
    outer_default: ClassVar[SweepValue] = SweepValue(start=0.0, stop=1.0, expts=21)
    filename_token: ClassVar[str] = "gain"

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot T1-with-tone, probe-gain sweep: at each probe gain it "
            "sweeps the wait-plus-probe length from both |g> and |e> initial "
            "states (Branch), classifies each shot in-program against the "
            "|g>/|e> centres, fits the dual transition rates per gain, and "
            "renders the 3×3 rate landscape over (gain, time). Runs on real "
            "hardware."
        ),
        expects_md=(
            "Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved "
            "cfg, not live MetaDict. Enter direct cfg values or optionally seed "
            "defaults with 'singleshot/ge' writeback. Missing or invalid cfg "
            "calibration fails before hardware. Optionally reads 'confusion_matrix' "
            "(readout correction) and 'ac_stark_coeff' (rescales the gain axis to photon "
            "number) at analyze time; 't1' to seed the length sweep stop; "
            "'readout_f' or 'r_f' plus 'res_ch' seed the probe tone."
        ),
        expects_ml=(
            "Needs a qubit pi-pulse module, a probe-tone pulse module, and a "
            "readout module. Optional reset (disabled when no library entry "
            "exists)."
        ),
        typical_writeback=(
            "No writeback — the rate landscape is read off the grid by eye."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'. "
            "Keep the gain sweep coarse (the inner "
            "length sweep multiplies the run time). Provide 'ac_stark_coeff' "
            "for a photon-number x-axis. 'uniform=True' (default) uses a linear "
            "length sweep; set False to cluster points along the expected "
            "exponential decay while preserving the configured window and count."
        ),
    )


class SsT1ToneSweepFreqAdapter(_SsT1ToneSweepBase):
    outer_key: ClassVar[str] = "freq"
    outer_label: ClassVar[str] = "Probe frequency (MHz)"
    outer_default: ClassVar[SweepValue] = SweepValue(
        start=5250.0, stop=6750.0, expts=21
    )
    filename_token: ClassVar[str] = "freq"

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot T1-with-tone, probe-frequency sweep: at each probe "
            "frequency it sweeps the wait-plus-probe length from both |g> and "
            "|e> initial states (Branch), classifies each shot in-program "
            "against the |g>/|e> centres, fits the dual transition rates per "
            "frequency, and renders the 3×3 rate landscape over (freq, time). "
            "Runs on real hardware."
        ),
        expects_md=(
            "Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved "
            "cfg, not live MetaDict. Enter direct cfg values or optionally seed "
            "defaults with 'singleshot/ge' writeback. Missing or invalid cfg "
            "calibration fails before hardware. Optionally reads 'confusion_matrix' "
            "(readout correction) at analyze time; 't1' to seed the length sweep stop; "
            "'readout_f' or 'r_f' plus 'rf_w' / 'res_ch' seed the probe drive "
            "and frequency sweep. "
            "('ac_stark_coeff' applies only to a gain sweep, not this one.)"
        ),
        expects_ml=(
            "Needs a qubit pi-pulse module, a probe-tone pulse module, and a "
            "readout module. Optional reset (disabled when no library entry "
            "exists)."
        ),
        typical_writeback=(
            "No writeback — the rate landscape is read off the grid by eye."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'. "
            "Keep the frequency sweep coarse (the "
            "inner length sweep multiplies the run time). 'uniform=True' "
            "(default) uses a linear length sweep; set False to cluster points "
            "along the expected exponential decay while preserving the configured "
            "window and count."
        ),
    )

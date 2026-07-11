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

from zcu_tools.experiment.v2.singleshot.t1 import (
    T1WithToneSweepCfg,
    T1WithToneSweepExp,
)
from zcu_tools.experiment.v2_gui.adapters._support import (
    FigureOnlyAnalyzeResult,
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    SweepDefault,
    custom,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    ExpContext,
    NoAnalyzeParams,
    RunRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CfgSchema,
    SweepValue,
)

from ._shared import read_ge_centers, readout_probe_freq, readout_probe_freq_range

SsT1ToneSweepRunResult: TypeAlias = Any  # T1WithToneSweepResult (frozen domain)


@dataclass
class SsT1ToneSweepAnalyzeResult(FigureOnlyAnalyzeResult):
    # The 3×3 rate landscape is look-at-the-grid: the domain analyze fits the
    # transition rates per outer-sweep point and renders them, returning only a
    # Figure. No writeback. ``figure`` is inherited.
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
        outer_expts = outer_default.expts
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
                    # Existing md_eval_scaled fallback was factor * 100.0.
                    stop=scaled_md("t1", factor=5.0, fallback_value=500.0),
                    expts=51,
                ),
            )
            .sweep(outer_key, label=outer_label, default=outer_sweep)
            .bool("uniform", label="Uniform (linear) sweep", default=True)
            .reps(1000)
            .rounds(10)
            .build()
        )

    def build_exp_cfg(
        self, raw_cfg: dict[str, object], req: RunRequest
    ) -> T1WithToneSweepCfg:
        # Pop ``uniform`` before lowering — it is a run-only flag.
        cfg_raw = dict(raw_cfg)
        cfg_raw.pop("uniform", None)
        return req.ml.make_cfg(cfg_raw, T1WithToneSweepCfg)

    def _uniform(self, raw_cfg: dict[str, object]) -> bool:
        value = raw_cfg.get("uniform", True)
        if not isinstance(value, bool):
            raise ValueError(f"'uniform' must be a bool, got {type(value).__name__}")
        return value

    def run(self, req: RunRequest, schema: CfgSchema) -> SsT1ToneSweepRunResult:
        # Override standard run: domain run needs GE centres + uniform kwarg.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        g_center, e_center, radius = read_ge_centers(req.md)
        uniform = self._uniform(raw_cfg)
        return T1WithToneSweepExp().run(
            soc, soccfg, cfg, g_center, e_center, radius, uniform=uniform
        )

    def analyze(
        self, req: AnalyzeRequest[SsT1ToneSweepRunResult, NoAnalyzeParams]
    ) -> SsT1ToneSweepAnalyzeResult:
        # ``ac_coeff`` (= md 'ac_stark_coeff') rescales the outer x-axis to photon
        # number; ``confusion_matrix`` readout-corrects; both are md inputs, not
        # user knobs. ``xlabel`` labels the rate plots with this adapter's outer
        # axis.
        ac_coeff = req.md.get("ac_stark_coeff")
        confusion = req.md.get("confusion_matrix")
        fig = T1WithToneSweepExp().analyze(
            req.run_result,
            ac_coeff=ac_coeff,
            confusion_matrix=confusion,
            xlabel=self.outer_label,
        )
        return SsT1ToneSweepAnalyzeResult(figure=fig)

    def make_filename_stem(self, ctx: ExpContext) -> str:
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
            "REQUIRES the single-shot discrimination calibration in the "
            "MetaDict — run 'singleshot/ge' first and apply its writeback so "
            "'g_center' / 'e_center' / 'ge_radius' are present; run fast-fails "
            "if any is missing. Optionally reads 'confusion_matrix' (readout "
            "correction) and 'ac_stark_coeff' (rescales the gain axis to photon "
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
            "Run after 'singleshot/ge'. Keep the gain sweep coarse (the inner "
            "length sweep multiplies the run time). Provide 'ac_stark_coeff' "
            "for a photon-number x-axis. 'uniform=True' (default) uses a linear "
            "length sweep; set False for log-spaced delays."
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
            "REQUIRES the single-shot discrimination calibration in the "
            "MetaDict — run 'singleshot/ge' first and apply its writeback so "
            "'g_center' / 'e_center' / 'ge_radius' are present; run fast-fails "
            "if any is missing. Optionally reads 'confusion_matrix' (readout "
            "correction) at analyze time; 't1' to seed the length sweep stop; "
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
            "Run after 'singleshot/ge'. Keep the frequency sweep coarse (the "
            "inner length sweep multiplies the run time). 'uniform=True' "
            "(default) uses a linear length sweep; set False for log-spaced "
            "delays."
        ),
    )

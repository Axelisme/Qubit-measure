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
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.singleshot.t1 import (
    T1WithToneSweepCfg,
    T1WithToneSweepExp,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    FigureOnlyAnalyzeResult,
    Init,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    md_eval_scaled,
    proper_relax,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    NoAnalyzeParams,
    RunRequest,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    require_soc_handles,
)

from ._shared import read_ge_centers

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
    def cfg_spec(cls) -> CfgSectionSpec:
        # ``uniform`` is a run-only flag (not lowered into T1WithToneSweepCfg). The
        # sweep section holds length + exactly one outer axis (gain XOR freq), so the
        # lowered cfg always has exactly one outer sweep set.
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "pi_pulse": make_pulse_module_spec(),
                "probe_pulse": make_pulse_module_spec(label="Probe Pulse"),
                "readout": make_readout_module_spec(),
            },
            sweep={
                "length": SweepSpec(label="Delay (us)"),
                cls.outer_key: SweepSpec(label=cls.outer_label),
            },
            extra={"uniform": ScalarSpec(label="Uniform (linear) sweep", type=bool)},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        sweep_stop = md_eval_scaled(ctx, "t1", factor=5.0, fallback=100.0)
        relax_delay = proper_relax(ctx)
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(
                reps=1000,
                rounds=10,
                relax_delay=relax_delay,
                # domain default is uniform=True (linear length sweep).
                uniform=True,
            )
            .role("modules.pi_pulse", "pi_pulse")
            .role("modules.probe_pulse", "qub_probe", Init.INLINE)
            .role("modules.readout", "readout")
            .role("modules.reset", "reset", Init.DISABLED)
            .set_sweep("sweep.length", SweepValue(start=0.0, stop=sweep_stop, expts=51))
            .set_sweep(f"sweep.{self.outer_key}", self.outer_default)
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
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
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
            "number) at analyze time; 't1' to seed the length sweep stop."
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
    outer_default: ClassVar[SweepValue] = SweepValue(start=-100.0, stop=100.0, expts=21)
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
            "correction) at analyze time; 't1' to seed the length sweep stop. "
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

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.reset.bath.freq import (
    FreqGainCfg,
    FreqGainExp,
    FreqGainResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    build_exp_spec,
    make_bath_reset_module_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    md_get_float,
    md_has_key,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    EvalValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    SaveDataRequest,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)

from ._shared import bath_reset_writeback_items

BathFreqGainRunResult: TypeAlias = FreqGainResult

# The pi/2 readout tomography pulse sits at a fixed phase offset; the domain
# adds a 4-point QickSweep1D("phase", 0, 270) on top of pi2_cfg.phase, so the
# form value is the *base* offset (notebook: phase=90), not a swept axis.
_PI2_PHASE_OFFSET_DEG: float = 90.0


@dataclass
class BathFreqGainAnalyzeParams:
    smooth: Annotated[float, ParamMeta(label="Smooth sigma", decimals=2)]


@dataclass
class BathFreqGainAnalyzeResult(AnalyzeResultBase):
    gain: float
    freq: float
    figure: Figure


class BathFreqGainAdapter(
    BaseAdapter[
        FreqGainCfg,
        BathFreqGainRunResult,
        BathFreqGainAnalyzeResult,
        BathFreqGainAnalyzeParams,
    ]
):
    exp_cls = FreqGainExp
    ExpCfg_cls: ClassVar[Any] = FreqGainCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Bath-reset cavity frequency–gain map: a 2D sweep of the bath "
                "reset's cavity-tone frequency × gain, with an internal 4-point "
                "pi/2 tomography phase axis the experiment drives itself, imaging "
                "the residual coherence to locate the (freq, gain) pair that best "
                "dumps the excitation into the cavity bath. Runs on real hardware."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional): 'r_f' / 'rabi_f' — the "
                "resonator frequency and bath Rabi frequency centring the cavity "
                "freq sweep at 'r_f - rabi_f' (absent → a fixed span); 'q_f' / "
                "'qub_ch' / 'res_ch' seed the bath-reset tone drive defaults."
            ),
            expects_ml=(
                "Needs a bath-reset module (the tested reset, with cavity / qubit / "
                "pi-2 tones) and a readout module. Optionally references a "
                "calibrated upstream reset and an init pulse (a library pi pulse "
                "when present) — both disabled when no library entry exists."
            ),
            typical_writeback=(
                "Proposes the best cavity gain into MetaDict 'bathreset_gain' and "
                "frequency into 'bathreset_freq'. No ModuleLibrary writeback — these "
                "seed the final 'reset_bath' registration done after the length and "
                "phase steps (D2(a))."
            ),
            recommended=(
                "Analysis smooths the 2D map before picking the peak; a 'smooth' "
                "sigma around 1 is a reasonable default — raise it on a noisy map. "
                "Saving is not offered: the experiment writes four phase-resolved "
                "files internally, which does not fit the single-file save path — "
                "read the optimum off the Analyze tab and write it back."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                # The two sweep axes own the cavity tone freq + gain
                # (set_param("res_freq"/"res_gain") at run); the internal phase
                # axis adds onto pi2_cfg.phase. Lock all three off the form so it
                # never shows a field the sweep silently overwrites (cavity
                # freq/gain → notebook 0.0; pi2 phase → notebook 90).
                "tested_reset": make_bath_reset_module_spec()
                .lock_literal("cavity_tone_cfg.freq", 0.0)
                .lock_literal("cavity_tone_cfg.gain", 0.0)
                .lock_literal("pi2_cfg.phase", _PI2_PHASE_OFFSET_DEG),
                "readout": make_readout_module_spec(),
            },
            sweep={
                "freq": SweepSpec(label="Cavity freq (MHz)"),
                "gain": SweepSpec(label="Cavity gain (a.u.)"),
            },
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=100, relax_delay=10.5)
            .role("modules.tested_reset", "bath_reset")
            .role("modules.readout", "readout", prefer_blank=True)
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .role("modules.init_pulse", "pi_pulse", optional=True)
            .set_sweep("sweep.freq", _cavity_freq_range(ctx, 51))
            .sweep("sweep.gain", 0.4, 1.0, 51)
            .build()
        )

    def get_analyze_params(
        self, result: BathFreqGainRunResult, ctx: ExpContext
    ) -> BathFreqGainAnalyzeParams:
        return BathFreqGainAnalyzeParams(smooth=1.0)

    def analyze(
        self, req: AnalyzeRequest[BathFreqGainRunResult, BathFreqGainAnalyzeParams]
    ) -> BathFreqGainAnalyzeResult:
        params = req.analyze_params
        gain, freq, fig = FreqGainExp().analyze(req.run_result, smooth=params.smooth)
        return BathFreqGainAnalyzeResult(gain=gain, freq=freq, figure=fig)

    def get_writeback_items(
        self,
        req: WritebackRequest[BathFreqGainRunResult, BathFreqGainAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            MetaDictWriteback(
                target_name="bathreset_gain",
                description="Bath-reset cavity gain (a.u.)",
                proposed_value=result.gain,
            ),
            MetaDictWriteback(
                target_name="bathreset_freq",
                description="Bath-reset cavity frequency (MHz)",
                proposed_value=result.freq,
            ),
        ]
        items.extend(bath_reset_writeback_items(req.ctx, req.run_result.cfg_snapshot))
        return items

    def save(self, req: SaveDataRequest[BathFreqGainRunResult]) -> None:
        # D3: the domain FreqGainExp.save writes four phase-resolved HDF5 files
        # (``<stem>_0deg`` .. ``<stem>_270deg``) and its load() takes a 4-path
        # list. The GUI save pipeline resolves a single ``.hdf5`` path up front
        # and reports it as written, so routing the multi-file save through it
        # would report a path that never exists (the display-vs-reality lie the
        # save layer guards against). Multi-file save is not in the framework's
        # contract, so this adapter does not support save — surface that plainly
        # rather than fake a single-file write or silently drop the data.
        del req
        raise NotImplementedError(
            "Bath freq–gain save is not supported in the GUI: the experiment "
            "writes four phase-resolved files, which the single-path save "
            "pipeline cannot represent. Read the optimum off the Analyze tab "
            "and write it back instead."
        )

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_bathreset_freqgain_{time.strftime('%m%d')}"


def _cavity_freq_range(ctx: ExpContext, expts: int) -> SweepValue:
    """Cavity-tone frequency sweep range: ``r_f - rabi_f`` centred.

    Mirrors the notebook span ``[r_f - 1.2*rabi_f, r_f - 0.8*rabi_f]``. When both
    md keys exist each edge stays an EvalValue (the GUI re-derives if md
    changes); otherwise a fixed fallback span around the available centre.
    """
    r_f = md_get_float(ctx, "r_f", 6000.0)
    rabi_f = md_get_float(ctx, "rabi_f", 20.0)
    if md_has_key(ctx, "r_f") and md_has_key(ctx, "rabi_f"):
        return SweepValue(
            start=EvalValue(expr="r_f - 1.2 * rabi_f"),
            stop=EvalValue(expr="r_f - 0.8 * rabi_f"),
            expts=expts,
        )
    return SweepValue(start=r_f - 1.2 * rabi_f, stop=r_f - 0.8 * rabi_f, expts=expts)

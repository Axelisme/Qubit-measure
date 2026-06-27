from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.v2.twotone.fluxdep import (
    FreqFluxCfg,
    FreqFluxExp,
    FreqFluxResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    FluxPickParams,
    FluxPickResult,
    build_exp_spec,
    build_flux_pick_session,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    proper_flux_range,
    proper_qub_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    ExpContext,
    InteractiveHost,
    InteractiveSession,
    MetaDictWriteback,
    RunRequest,
    SweepSpec,
    WritebackItem,
    WritebackRequest,
)

FluxDepRunResult: TypeAlias = FreqFluxResult


class FluxDepAdapter(
    BaseAdapter[FreqFluxCfg, FluxDepRunResult, FluxPickResult, FluxPickParams]
):
    exp_cls = FreqFluxExp
    legacy_migration_experiment: ClassVar[str | None] = "twotone/flux_dep"
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.INTERACTIVE
    )

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Two-tone qubit flux dependence: a 2D scan stepping an external "
                "flux device (outer) against the qubit-drive frequency (inner), "
                "reading out the resonator at each point, to map the qubit "
                "spectrum versus flux (the Fluxonium flux-dispersion arcs). Runs "
                "on real hardware and needs a connected flux device. No automated "
                "fit in the GUI — points are extracted interactively. This is not "
                "the early flux-finder; use it only after readout and qubit "
                "spectroscopy are already credible."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional): 'q_f' — qubit frequency, "
                "centring the inner sweep (~2000–6000 MHz); 'qf_w' — qubit "
                "linewidth, half-span ±qf_w (~1–50 MHz); 'qub_ch' — qubit-drive "
                "channel; 'r_f' — resonator frequency for the readout tone "
                "(~4000–8000 MHz); 'res_ch' / 'ro_ch' — readout drive / ADC "
                "channels; 'timeFly' — readout trigger offset (~0–1 us); "
                "'flx_half' / 'flx_int' — calibrated half-flux / integer-flux "
                "device values that set the flux bounds (device-specific units; "
                "absent → fixed [-4e-3, 4e-3])."
            ),
            expects_ml=(
                "Needs a qubit-probe pulse module and a pulse-readout module; "
                "references a ModuleLibrary waveform named 'ro_waveform' when "
                "present. Optionally references a calibrated reset module. Also "
                "needs a flux device reference 'flux_dev' (a connected device, "
                "default 'flux_yoko') — a device, not a library entry."
            ),
            typical_writeback=(
                "Interactive analysis (not a fit): after the run, the user drags "
                "two lines on the 2D map to mark the half-flux and integer-flux "
                "sweet spots, then clicks Done. The result writes back 'flx_half', "
                "'flx_int' and 'flx_period' (= 2·|flx_int − flx_half|) to the "
                "MetaDict as a preview. Picking is user/agent judgement from the "
                "measured map; do not use this preview to replace the earlier "
                "onetone/flux_dep flux calibration unless the map is clearly "
                "trustworthy."
            ),
            recommended=(
                "No fit options. Use after onetone/flux_dep has established the "
                "flux period and twotone/freq has found 'q_f'. Do not run this "
                "early to find flux: with uncalibrated readout or an unknown "
                "qubit-drive window the arc is usually invisible. Default sweep: "
                "flux ~101 points across roughly one period (bracketed by "
                "'flx_int'/'flx_half' when calibrated), frequency ~101 points "
                "spanning ±qf_w around 'q_f'. For a later model survey use a wide "
                "flux range and a broad frequency window at moderate drive gain "
                "to capture the full arc, then narrow both. "
                "For consecutive flux sweeps, reverse the sweep direction (swap "
                "start/stop) on the next run so the flux source need not ramp "
                "back across the full range first. Inspect the 2D map with "
                "gui_tab_get_current_figure. Confirm 'flux_dev' points at a "
                "connected device before running."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                # The sweep axis owns the qubit-drive frequency
                # (set_param("freq") at run); lock it so the form does not
                # show a field the sweep silently overwrites.
                "qub_pulse": make_pulse_module_spec().lock_literal("freq", 0.0),
                "readout": make_readout_module_spec(),
            },
            dev={"flux_dev": DeviceRefSpec(label="Flux Device")},
            sweep={
                "flux": SweepSpec(label="Flux device value", decimals=6),
                "freq": SweepSpec(label="Freq (MHz)"),
            },
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=100, relax_delay=1.0)
            # optional → None (disabled) when no library reset (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .role("modules.qub_pulse", "qub_probe", prefer_blank=True)
            .role("modules.readout", "readout")
            .set("dev.flux_dev", "flux_yoko")
            .set_sweep("sweep.flux", proper_flux_range(ctx, 101))
            .set_sweep("sweep.freq", proper_qub_freq_range(ctx, 1001, span_factor=1.0))
            .build()
        )

    def get_analyze_params(
        self, result: FluxDepRunResult, ctx: ExpContext
    ) -> FluxPickParams:
        del result, ctx
        # No tunable analyze params; the magnitude-only projection is hardcoded
        # in setup_interactive_analysis (see below).
        return FluxPickParams()

    def setup_interactive_analysis(
        self,
        req: AnalyzeRequest[FluxDepRunResult, FluxPickParams],
        host: InteractiveHost,
    ) -> InteractiveSession:
        # Two-tone qubit spectra may carry useful phase information, so the
        # magnitude-only projection is fixed False (not surfaced as an analyze param).
        return build_flux_pick_session(req, host, force_magnitude=False)

    def get_writeback_items(
        self, req: WritebackRequest[FluxDepRunResult, FluxPickResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="flx_half",
                description="Half-flux (Φ₀/2) sweet-spot device value",
                proposed_value=result.flx_half,
            ),
            MetaDictWriteback(
                target_name="flx_int",
                description="Integer-flux sweet-spot device value",
                proposed_value=result.flx_int,
            ),
            MetaDictWriteback(
                target_name="flx_period",
                description="Flux period (device units) = 2·|flx_int − flx_half|",
                proposed_value=result.flx_period,
            ),
        ]

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FreqFluxCfg:
        cfg_raw = dict(raw_cfg)
        dev_raw = cfg_raw.pop("dev")
        if not isinstance(dev_raw, dict):
            raise RuntimeError("FluxDep dev section must lower to a dict")
        dev_patch: dict[str, dict] = {}
        for label_key, device_name in dev_raw.items():
            if not isinstance(device_name, str) or not device_name:
                raise RuntimeError(
                    f"FluxDep dev.{label_key} must be a non-empty device name"
                )
            dev_patch[device_name] = {"label": label_key}
        cfg_raw["dev"] = dev_patch
        return req.ml.make_cfg(cfg_raw, FreqFluxCfg)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_qubit_flux"

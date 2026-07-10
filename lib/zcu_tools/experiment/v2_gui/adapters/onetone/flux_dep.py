from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.v2.onetone.flux_dep import (
    FluxDepCfg,
    FluxDepExp,
    FluxDepResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    FluxPickParams,
    FluxPickResult,
    Init,
    build_exp_spec,
    build_flux_pick_session,
    make_pulse_readout_module_spec,
    md_get_float,
    md_has_key,
    proper_flux_range,
    proper_res_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    CfgSectionSpec,
    CfgSectionValue,
    EvalValue,
    ExpContext,
    InteractiveHost,
    InteractiveSession,
    MetaDictWriteback,
    RunRequest,
    ScalarSpec,
    SweepSpec,
    WritebackItem,
    WritebackRequest,
)

OneToneFluxDepRunResult: TypeAlias = FluxDepResult


class OneToneFluxDepAdapter(
    BaseAdapter[FluxDepCfg, OneToneFluxDepRunResult, FluxPickResult, FluxPickParams]
):
    exp_cls = FluxDepExp
    legacy_migration_experiment: ClassVar[str | None] = "onetone/flux_dep"
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.INTERACTIVE
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "One-tone resonator flux dependence: a 2D sweep of an external "
            "flux-bias device value versus readout frequency, tracing how the "
            "resonator frequency moves with flux. The basis for locating a "
            "Fluxonium's flux sweet spots (half-flux and integer-flux points). "
            "Runs on real hardware; requires a SoC connection and a configured "
            "flux-bias device."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'r_f' — resonator "
            "frequency, centring the frequency sweep (~4000–8000 MHz); 'rf_w' "
            "— linewidth, span r_f ± rf_w (~5–50 MHz; falls back to ±30 MHz); "
            "'res_probe_len' — readout probe length, from which ro_length is "
            "derived as 'res_probe_len - 0.1' us (~0.5–5 us); 'res_ch' / "
            "'ro_ch' — drive / ADC channels; 'timeFly' — trigger-offset cable "
            "delay; 'flx_half' / 'flx_int' — already-calibrated half-flux / "
            "integer-flux device values when present. First bring-up must not "
            "assume these are known; absent values fall back to a fixed "
            "[-4e-3, 4e-3] device sweep that the operator should adjust to a "
            "safe, user-approved range."
        ),
        expects_ml=(
            "Needs a pulse-readout module, and references a ModuleLibrary "
            "waveform named 'ro_waveform' when present (optional)."
        ),
        typical_writeback=(
            "Interactive analysis (not a fit): after the run, the user drags "
            "two lines on the 2D map to mark the half-flux and integer-flux "
            "sweet spots, then clicks Done. The result writes back 'flx_half', "
            "'flx_int', and 'flx_period' (= 2·|flx_int − flx_half|) to the "
            "MetaDict as a preview. The picking is user/agent judgement from "
            "the measured map, not simulator truth; apply writeback only after "
            "reviewing the figure."
        ),
        recommended=(
            "Run after lookback and an initial onetone/freq, before any "
            "twotone flux mapping, to find 'flx_int' / period from the "
            "resonator map. Interactive flux-line pick (no fit). Also set the "
            "'flux_dev' field — the flux-bias device reference (defaults to a "
            "registered device named 'flux', falling back to 'flux_yoko') — "
            "and confirm it points at a connected device. Typical sweep: "
            "flux ~101 points "
            "across one period (driven by flx_half/flx_int when calibrated), "
            "frequency r_f ± one linewidth over ~101 points. Start at a low "
            "readout gain (~0.005) to stay below punch-out so the dip tracks "
            "cleanly; if the dip is hard to see (poor SNR), raise the gain "
            "toward ~0.05 while keeping the frequency window tight. Survey "
            "wide, then narrow around a sweet spot — and if the resonator's "
            "flux dispersion (dispersive shift) looks small, narrow the "
            "frequency window early instead of keeping a broad default span. "
            "For consecutive flux sweeps, reverse the sweep direction (swap "
            "start/stop) on the next run so the current source need not ramp "
            "all the way back across the full range first. Inspect the 2D map "
            "with gui_tab_get_current_figure to judge window / SNR / shift. "
            "After accepting 'flx_int', move the flux device there and re-run "
            "onetone/freq at that flux before twotone/freq."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                # No reset module — one-tone runs without a qubit reset
                # (the ExpCfg defaults reset=None). The freq sweep owns
                # the readout frequency (set_param("freq") writes both
                # pulse and ro freq), so lock it off the form.
                "readout": make_pulse_readout_module_spec()
                .lock_literal("pulse_cfg.freq", 0.0)
                .lock_literal("ro_cfg.ro_freq", 0.0),
            },
            dev={
                "flux_dev": ScalarSpec(
                    label="Flux Device",
                    type=str,
                    choices_source="devices",
                    required=True,
                )
            },
            sweep={
                "flux": SweepSpec(label="Flux device value", decimals=6),
                "freq": SweepSpec(label="Freq (MHz)"),
            },
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        probe_len = md_get_float(ctx, "res_probe_len", 1.0)
        ro_length: float | EvalValue = (
            EvalValue(expr="res_probe_len - 0.1")
            if md_has_key(ctx, "res_probe_len") and probe_len > 0.1
            else probe_len - 0.1
        )
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=1, relax_delay=1.0)
            .role("modules.readout", "readout", Init.INLINE)
            .set("modules.readout.pulse_cfg.gain", 0.05)
            .set("modules.readout.ro_cfg.ro_length", ro_length)
            .value_ref("dev.flux_dev", "device.flux.name", default="flux_yoko")
            .set_sweep("sweep.flux", proper_flux_range(ctx, 101))
            .set_sweep("sweep.freq", proper_res_freq_range(ctx, 101, span_factor=1.0))
            .build()
        )

    # -- interactive analysis: user picks the half/integer flux lines ----------

    def setup_interactive_analysis(
        self,
        req: AnalyzeRequest[OneToneFluxDepRunResult, FluxPickParams],
        host: InteractiveHost,
    ) -> InteractiveSession:
        # One-tone resonator spectra are magnitude-only (phase is uninformative),
        # so the projection is fixed True and not surfaced as an analyze param.
        return build_flux_pick_session(req, host, force_magnitude=True)

    def get_writeback_items(
        self, req: WritebackRequest[OneToneFluxDepRunResult, FluxPickResult]
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

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FluxDepCfg:
        cfg_raw = dict(raw_cfg)
        dev_raw = cfg_raw.pop("dev")
        if not isinstance(dev_raw, dict):
            raise RuntimeError("FluxDep dev section must lower to a dict")
        # dev_raw = {"flux_dev": "flux_yoko", ...}
        # convert to make_cfg patch format: {"flux_yoko": {"label": "flux_dev"}}
        dev_patch: dict[str, dict] = {}
        for label_key, device_name in dev_raw.items():
            if not isinstance(device_name, str) or not device_name:
                raise RuntimeError(
                    f"FluxDep dev.{label_key} must be a non-empty device name"
                )
            dev_patch[device_name] = {"label": label_key}
        cfg_raw["dev"] = dev_patch
        return req.ml.make_cfg(cfg_raw, FluxDepCfg)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_flux"

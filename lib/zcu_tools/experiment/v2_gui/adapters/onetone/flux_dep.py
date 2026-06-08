from __future__ import annotations

from typing_extensions import ClassVar, TypeAlias, Union

from zcu_tools.experiment.v2.onetone.flux_dep import (
    FluxDepCfg,
    FluxDepExp,
    FluxDepResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    build_exp_spec,
    make_pulse_readout_module_spec,
    md_get_float,
    md_has_key,
    proper_flux_range,
    proper_res_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    EvalValue,
    ExpContext,
    RunRequest,
    SweepSpec,
)

OneToneFluxDepRunResult: TypeAlias = FluxDepResult


class OneToneFluxDepAdapter(BaseAdapter[FluxDepCfg, OneToneFluxDepRunResult]):
    exp_cls = FluxDepExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=True, supports_analysis=False
    )

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
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
                "delay; 'flx_half' / 'flx_int' — previously calibrated half-flux / "
                "integer-flux device values that set the flux sweep to span ~one "
                "period (device-specific units; absent → fixed [-4e-3, 4e-3])."
            ),
            expects_ml=(
                "Needs a pulse-readout module, and references a ModuleLibrary "
                "waveform named 'ro_waveform' when present (optional)."
            ),
            typical_writeback=(
                "No writeback — no automated analysis. The underlying experiment "
                "opens an interactive line-picker for the user to mark the flux "
                "points by hand; the chosen points are recorded manually "
                "elsewhere, not through this adapter."
            ),
            recommended=(
                "No automated analysis. Also set the 'flux_dev' field — the "
                "flux-bias device reference (default 'flux_yoko') — and confirm it "
                "points at a connected device. Typical sweep: flux ~101 points "
                "across one period (driven by flx_half/flx_int when calibrated), "
                "frequency r_f ± one linewidth over ~101 points, at a low readout "
                "gain (~0.005) to stay below punch-out so the dip tracks cleanly. "
                "Survey wide, then narrow around a sweet spot."
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
            dev={"flux_dev": DeviceRefSpec(label="Flux Device")},
            sweep={
                "flux": SweepSpec(label="Flux device value", decimals=6),
                "freq": SweepSpec(label="Freq (MHz)"),
            },
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        probe_len = md_get_float(ctx, "res_probe_len", 1.0)
        ro_length: Union[float, EvalValue] = (
            EvalValue(expr="res_probe_len - 0.1")
            if md_has_key(ctx, "res_probe_len")
            else probe_len - 0.1
        )
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=1, relax_delay=1.0)
            .role("modules.readout", "readout", prefer_blank=True)
            .set("modules.readout.pulse_cfg.gain", 0.005)
            .set("modules.readout.ro_cfg.ro_length", ro_length)
            .set("dev.flux_dev", "flux_yoko")
            .set_sweep("sweep.flux", proper_flux_range(ctx, 101))
            .set_sweep("sweep.freq", proper_res_freq_range(ctx, 101, span_factor=1.0))
            .build()
        )

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

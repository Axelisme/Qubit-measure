from __future__ import annotations

from typing_extensions import ClassVar, TypeAlias

from zcu_tools.experiment.v2.twotone.fluxdep import (
    FreqFluxCfg,
    FreqFluxExp,
    FreqFluxResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_qub_probe_default,
    make_readout_default,
    make_readout_module_spec,
    make_reset_module_spec,
    make_reset_ref_default,
    proper_flux_range,
    proper_qub_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    CfgNodeValue,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    ExpContext,
    RunRequest,
    ScalarSpec,
    SweepSpec,
)

FluxDepRunResult: TypeAlias = FreqFluxResult


class FluxDepAdapter(BaseAdapter[FreqFluxCfg, FluxDepRunResult]):
    exp_cls = FreqFluxExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=True, supports_analysis=False
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
                "fit in the GUI — points are extracted interactively."
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
                "No GUI analysis and no automatic writeback. The underlying "
                "experiment offers interactive tools (used in notebooks) to pick "
                "the flux arcs and the 'flx_half'/'flx_int' points by hand; those "
                "are not wired through this adapter."
            ),
            recommended=(
                "No fit options. Default sweep: flux ~101 points across roughly "
                "one period (bracketed by 'flx_int'/'flx_half' when calibrated), "
                "frequency ~101 points spanning ±qf_w around 'q_f'. For a first "
                "survey use a wide flux range and a broad frequency window at "
                "moderate drive gain to capture the full arc, then narrow both. "
                "Confirm 'flux_dev' points at a connected device before running."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "reset": make_reset_module_spec(optional=True),
                        "qub_pulse": make_pulse_module_spec(),
                        "readout": make_readout_module_spec(),
                    },
                ),
                "dev": CfgSectionSpec(
                    label="Flux Device",
                    fields={
                        "flux_dev": DeviceRefSpec(label="Flux Device"),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={
                        "flux": SweepSpec(label="Flux device value", decimals=6),
                        "freq": SweepSpec(label="Freq (MHz)"),
                    },
                ),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        _module_fields: dict[str, CfgNodeValue] = {
            "qub_pulse": make_qub_probe_default(ctx),
            "readout": make_readout_default(ctx),
            # optional → DisabledRefValue when no library reset (ADR-0012)
            "reset": make_reset_ref_default(ctx, optional=True),
        }
        return CfgSectionValue(
            fields={
                "modules": CfgSectionValue(fields=_module_fields),
                "dev": CfgSectionValue(
                    fields={
                        "flux_dev": DirectValue("flux_yoko"),
                    }
                ),
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
                "relax_delay": DirectValue(1.0),
                "sweep": CfgSectionValue(
                    fields={
                        "flux": proper_flux_range(ctx, 101),
                        "freq": proper_qub_freq_range(ctx, 101, span_factor=1.0),
                    }
                ),
            }
        )

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

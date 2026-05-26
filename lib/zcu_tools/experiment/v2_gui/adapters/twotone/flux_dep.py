from __future__ import annotations

from typing_extensions import TypeAlias

from zcu_tools.experiment.v2.twotone.fluxdep import (
    FreqFluxCfg,
    FreqFluxExp,
    FreqFluxResult,
)
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_pulse_ref_default,
    make_readout_default,
    make_readout_module_spec,
    make_reset_module_spec,
    make_reset_ref_default,
    md_get_float,
)
from zcu_tools.gui.adapter import (
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    ExpContext,
    NoAnalysisAdapterMixin,
    RunRequest,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)

FluxDepRunResult: TypeAlias = FreqFluxResult


class FluxDepAdapter(NoAnalysisAdapterMixin[FreqFluxCfg, FluxDepRunResult]):
    exp_cls = FreqFluxExp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        q_f = md_get_float(ctx, "q_f", 4000.0)
        qf_w = md_get_float(ctx, "qf_w", 20.0)
        half_span = qf_w if qf_w > 0 else 20.0
        root_spec = CfgSectionSpec(
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
        _module_fields: dict[str, CfgNodeValue] = {
            "qub_pulse": make_pulse_ref_default(ctx),
            "readout": make_readout_default(ctx),
        }
        _reset = make_reset_ref_default(ctx, optional=True)
        if _reset is not None:
            _module_fields["reset"] = _reset
        root_val = CfgSectionValue(
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
                        "flux": SweepValue(start=3.57e-3, stop=3.61e-3, expts=101),
                        "freq": SweepValue(
                            start=q_f - half_span,
                            stop=q_f + half_span,
                            expts=101,
                        ),
                    }
                ),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

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

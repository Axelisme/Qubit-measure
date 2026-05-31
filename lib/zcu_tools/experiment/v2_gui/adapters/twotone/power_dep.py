from __future__ import annotations

import time

from typing_extensions import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.twotone.power_dep import PowerCfg, PowerExp, PowerResult
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_qub_probe_default,
    make_readout_default,
    make_readout_module_spec,
    make_reset_module_spec,
    make_reset_ref_default,
    proper_qub_freq_range,
)
from zcu_tools.gui.adapter import (
    AdapterCapabilities,
    CfgNodeValue,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)

PowerDepRunResult: TypeAlias = PowerResult


class PowerDepAdapter(BaseAdapter[PowerCfg, PowerDepRunResult]):
    exp_cls = PowerExp
    ExpCfg_cls: ClassVar[Any] = PowerCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=True, supports_analysis=False
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
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={
                        "gain": SweepSpec(label="Gain (a.u.)"),
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
        root_val = CfgSectionValue(
            fields={
                "modules": CfgSectionValue(fields=_module_fields),
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
                "relax_delay": DirectValue(1.0),
                "sweep": CfgSectionValue(
                    fields={
                        "gain": SweepValue(start=0.001, stop=0.5, expts=101),
                        "freq": proper_qub_freq_range(ctx, 201),
                    }
                ),
            }
        )
        return root_val

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_qubit_power_{time.strftime('%H%M')}"

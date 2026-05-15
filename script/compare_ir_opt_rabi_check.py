from __future__ import annotations

from typing import Any

import zcu_tools.experiment.v2 as ze
from compare_ir_opt_common import (
    ExperimentCase,
    MetaDict,
    ModuleLibrary,
    QickConfig,
    parse_args,
    run_case,
)
from zcu_tools.notebook.utils import make_sweep


def build_rabi_check_cfg(
    ml: ModuleLibrary, _md: MetaDict
) -> ze.twotone.reset.RabiCheckCfg:
    exp_cfg = {
        "modules": {
            "reset": "reset_bath",
            "rabi_pulse": "pi_amp",
            "tested_reset": "reset_bath",
            "pi_pulse": "pi_amp",
            "readout": "readout_dpm",
        },
        "sweep": {"gain": make_sweep(0.0, 1.0, 21)},
        "relax_delay": 0.5,
    }
    return ml.make_cfg(exp_cfg, ze.twotone.reset.RabiCheckCfg, reps=100, rounds=1)


def run_rabi_check(soc: Any, soccfg: QickConfig, cfg: Any, _ml: ModuleLibrary) -> Any:
    return ze.twotone.reset.RabiCheckExp().run(soc, soccfg, cfg)


def main() -> None:
    case = ExperimentCase(
        name="rabi_check",
        build_cfg=build_rabi_check_cfg,
        run_experiment=run_rabi_check,
    )
    args = parse_args(case.name)
    run_case(case, args)


if __name__ == "__main__":
    main()

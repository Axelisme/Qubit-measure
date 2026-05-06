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


def build_zigzag_sweep_cfg(
    ml: ModuleLibrary, _md: MetaDict
) -> ze.twotone.ZigZagScanCfg:
    exp_cfg = {
        "modules": {
            "reset": "reset_bath",
            "X90_pulse": "pi2_amp",
            "X180_pulse": "pi_amp",
            "readout": "readout_dpm",
        },
        "sweep": {"gain": make_sweep(0.0, 1.0, 21)},
        "n_times": 10,
        "relax_delay": 0.5,
    }
    return ml.make_cfg(exp_cfg, ze.twotone.ZigZagScanCfg, reps=100, rounds=1)


def run_zigzag_sweep(soc: Any, soccfg: QickConfig, cfg: Any, _ml: ModuleLibrary) -> Any:
    return ze.twotone.ZigZagScanExp().run(soc, soccfg, cfg)


def main() -> None:
    case = ExperimentCase(
        name="zigzag_sweep",
        build_cfg=build_zigzag_sweep_cfg,
        run_experiment=run_zigzag_sweep,
    )
    args = parse_args(case.name)
    run_case(case, args)


if __name__ == "__main__":
    main()

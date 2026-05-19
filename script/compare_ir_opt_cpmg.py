from __future__ import annotations

from compare_ir_opt_common import (
    ExperimentCase,
    build_cpmg_cfg,
    parse_args,
    run_case,
    run_cpmg,
)


def main() -> None:
    case = ExperimentCase(
        name="cpmg",
        build_cfg=build_cpmg_cfg,
        run_experiment=run_cpmg,
    )
    args = parse_args(case.name)
    run_case(case, args)


if __name__ == "__main__":
    main()

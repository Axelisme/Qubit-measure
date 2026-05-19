from __future__ import annotations

from compare_ir_opt_common import (
    ExperimentCase,
    build_allxy_cfg,
    parse_args,
    run_allxy,
    run_case,
)


def main() -> None:
    case = ExperimentCase(
        name="allxy",
        build_cfg=build_allxy_cfg,
        run_experiment=run_allxy,
    )
    args = parse_args(case.name)
    run_case(case, args)


if __name__ == "__main__":
    main()

from __future__ import annotations

from compare_ir_opt_common import (
    ExperimentCase,
    build_rb_cfg,
    parse_args,
    run_case,
    run_rb,
)


def main() -> None:
    case = ExperimentCase(
        name="rb",
        build_cfg=build_rb_cfg,
        run_experiment=run_rb,
    )
    args = parse_args(case.name)
    run_case(case, args)


if __name__ == "__main__":
    main()

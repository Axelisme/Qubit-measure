"""Build optional plateau data for the adjacent FTF environment YAML.

Pass --duration-ns, --ratios, --lower and --upper. --degree selects 4 or 6;
--node-limit permits partial work. This entry executes the importable builder,
not a generated calculation script. The Intel OpenCL gpu extra is required.
"""

import sys
from pathlib import Path

from zcu_tools.experiment.ftf.database import main

if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    raise SystemExit(
        main(
            [
                str(folder / "ftf.yaml"),
                str(folder / "evolution_data"),
                "--compiler-directory",
                str(folder / "evolution_data" / "compiler"),
                *sys.argv[1:],
            ]
        )
    )

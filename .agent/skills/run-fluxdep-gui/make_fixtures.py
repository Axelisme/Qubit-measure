#!/usr/bin/env python
"""Generate the smoke fixtures from the raw Database/Q3_2D/ spectra.

The raw files store their axes transposed (x=freq Hz, y=flux, z=(freq, flux))
versus what fluxdep-gui's loader expects (x=flux, y=freq Hz). This re-saves them
in the canonical layout into ``fixtures/`` next to this script.

Both the raw files and the .hdf5 fixtures are gitignored (large + Database/), so
run this once after a fresh checkout that has the Database files:

    .venv/bin/python .claude/skills/run-fluxdep-gui/make_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent

JOBS = [
    ("Database/Q3_2D/Q2/s002_onetone_flux_Q2_2.hdf5", "onetone_flux_Q2"),
    ("Database/Q3_2D/Q1/003_qubit_flux_spec_ge_Q1_1.hdf5", "twotone_flux_Q1"),
]


def main() -> int:
    from zcu_tools.utils.datasaver import load_data, save_data

    (HERE / "fixtures").mkdir(exist_ok=True)
    for rel, name in JOBS:
        src = REPO / rel
        if not src.exists():
            print(f"[make_fixtures] missing raw file: {src}", file=sys.stderr)
            return 2
        z, freq_hz, flux = load_data(str(src), return_comment=False)  # z=(freq,flux)
        if flux is None:
            print(f"[make_fixtures] {src} is not a 2D sweep", file=sys.stderr)
            return 2
        signals2d = z.T  # (flux, freq)
        save_data(
            filepath=str(HERE / "fixtures" / name),
            x_info={"name": "Flux device value", "unit": "a.u.", "values": flux.astype(np.float64)},
            y_info={"name": "Frequency", "unit": "Hz", "values": freq_hz.astype(np.float64)},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2d.T},
        )
        print(f"[make_fixtures] wrote fixtures/{name}_1.hdf5")
    return 0


if __name__ == "__main__":
    sys.exit(main())

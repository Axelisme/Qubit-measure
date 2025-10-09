from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    set_reset_cfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real

from ....runner import HardTask, Runner

# (fpts, phases, signals_2d)
LengthResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def bathreset_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class LengthExperiment(AbsExperiment[LengthResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
        detune: float = 0.0,
    ) -> LengthResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        if cfg["tested_reset"]["type"] != "bath":
            raise ValueError("This experiment only supports bath reset")

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        lens = sweep2array(cfg["sweep"]["length"])  # predicted frequency points

        len_spans = sweep2param("length", cfg["sweep"]["length"])
        set_reset_cfg(cfg["tested_reset"], "length", len_spans)
        set_reset_cfg(cfg["tested_reset"], "pi2_phase", 360 * detune * len_spans)

        with LivePlotter1D(
            "Length (us)", "Signal (a.u.)", disable=not progress
        ) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                make_reset("reset", ctx.cfg.get("reset")),
                                Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                                make_reset("tested_reset", ctx.cfg["tested_reset"]),
                                make_readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(lens),),
                ),
                update_hook=lambda ctx: viewer.update(
                    lens, bathreset_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (lens, signals)

        return lens, signals

    def analyze(self, result: Optional[LengthResultType] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        raise NotImplementedError("Analysis not implemented yet")

    def save(
        self,
        filepath: str,
        result: Optional[LengthResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/length",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Length", "unit": "s", "values": lens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

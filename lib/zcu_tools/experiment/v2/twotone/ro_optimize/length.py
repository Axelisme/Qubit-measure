from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typing_extensions import NotRequired
from matplotlib.figure import Figure

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, make_ge_sweep, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.experiment.v2.utils import snr_as_signal
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

LengthResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


class OptimizeLengthTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class OptimizeLengthExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: OptimizeLengthTaskConfig) -> LengthResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        length_sweep = cfg["sweep"]["length"]

        # replace length sweep with ge sweep, and use soft loop for length
        cfg["sweep"] = {"ge": make_ge_sweep()}

        # set with / without pi gain for qubit pulse
        Pulse.set_param(
            cfg["qub_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        lengths = sweep2array(length_sweep)  # predicted readout lengths

        # set initial readout length and adjust pulse length
        Readout.set_param(
            cfg["readout"], "ro_length", sweep2param("length", length_sweep)
        )
        Readout.set_param(cfg["readout"], "length", lengths.max() + 0.1)

        with LivePlotter1D("Readout Length (us)", "SNR") as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="length",
                    sweep_values=lengths.tolist(),
                    update_cfg_fn=lambda _, ctx, length: Readout.set_param(
                        ctx.cfg["readout"], "ro_length", length
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            (
                                prog := ModularProgramV2(
                                    soccfg,
                                    ctx.cfg,
                                    modules=[
                                        Reset(
                                            "reset",
                                            ctx.cfg.get("reset", {"type": "none"}),
                                        ),
                                        Pulse("qub_pulse", ctx.cfg["qub_pulse"]),
                                        Readout("readout", ctx.cfg["readout"]),
                                    ],
                                )
                            )
                            and (
                                prog.acquire(
                                    soc,
                                    progress=False,
                                    callback=update_hook,
                                    record_statistic=True,
                                ),
                                prog.get_covariance(),
                                prog.get_median(),
                            )
                        ),
                        raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    lengths, np.abs(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

        # record the last cfg and result
        self.last_cfg = cfg
        self.last_result = (lengths, signals)

        return lengths, signals

    def analyze(
        self, result: Optional[LengthResultType] = None, *, t0: Optional[float] = None
    ) -> Tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lengths, signals = result

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, 1)

        if t0 is None:
            max_id = np.argmax(snrs)
        else:
            max_id = np.argmax(snrs / np.sqrt(lengths + t0))

        max_length = float(lengths[max_id])
        max_snr = float(snrs[max_id])

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(lengths, snrs)
        ax.axvline(max_length, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Readout Length (us)")
        ax.set_ylabel("SNR (a.u.)")
        ax.legend()
        ax.grid(True)

        return max_length, fig

    def save(
        self,
        filepath: str,
        result: Optional[LengthResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/length",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lengths, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Readout Length", "unit": "s", "values": lengths * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> LengthResultType:
        signals, lengths, _ = load_data(filepath, **kwargs)
        assert lengths is not None
        assert len(lengths.shape) == 1 and len(signals.shape) == 1
        assert lengths.shape == signals.shape

        lengths = lengths * 1e6  # s -> us

        lengths = lengths.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (lengths, signals)

        return lengths, signals

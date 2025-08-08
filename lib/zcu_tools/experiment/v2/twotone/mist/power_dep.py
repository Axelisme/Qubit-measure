import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from tqdm.auto import tqdm

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data

from ...template import sweep_hard_template

MISTPowerDepResultType = Tuple[np.ndarray, np.ndarray]


class MISTPowerDep(AbsExperiment[MISTPowerDepResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress=True
    ) -> MISTPowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        qub_pulse = cfg["qub_pulse"]

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        pdr_sweep = cfg["sweep"]["gain"]

        qub_pulse["gain"] = sweep2param("gain", pdr_sweep)

        pdrs = sweep2array(pdr_sweep)  # predicted amplitudes

        prog = TwoToneProgram(soccfg, cfg)

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D("Pulse gain", "MIST", disable=not progress),
            ticks=(pdrs,),
            catch_interrupt=progress,
        )

        # get the actual amplitudes
        pdrs = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)  # type: ignore
        assert isinstance(pdrs, np.ndarray), "pdrs should be an array"

        # record the last result
        self.last_cfg = cfg
        self.last_result = (pdrs, signals)

        return pdrs, signals

    def analyze(
        self,
        result: Optional[MISTPowerDepResultType] = None,
        *,
        g0=None,
        e0=None,
        ac_coeff=None,
    ) -> None:
        if result is None:
            result = self.last_result

        pdrs, signals = result

        if g0 is None:
            g0 = signals[0]

        amp_diff = np.abs(signals - g0)

        if ac_coeff is None:
            xs = pdrs
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * pdrs**2
            xlabel = r"$\bar n$"

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.plot(xs, amp_diff.T, marker=".")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))

    def save(
        self,
        filepath: str,
        result: Optional[MISTPowerDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/mist/pdr",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )


MISTPowerDepOvernightResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


class MISTPowerDepOvernight(AbsExperiment[MISTPowerDepOvernightResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress=True
    ) -> MISTPowerDepOvernightResultType:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

        dh = display(fig, display_id=True)

        total_time = 1 * 60 * 60  # 1 hours in seconds
        interval = 5 * 60  # 5 minutes in seconds

        mist_pdr_exp = MISTPowerDep()

        iters = list(range(total_time // interval))
        overnight_signals = []
        try:
            for i in tqdm(iters, desc="Overnight Scans", unit="iteration"):
                start_t = time.time()

                pdrs, signals = mist_pdr_exp.run(soc, soccfg, cfg, progress=False)
                overnight_signals.append(signals)

                signals_array = np.array(overnight_signals)
                g0 = np.mean(signals_array[:, 0])

                # Left plot: Current scan
                ax_left.clear()
                ax_left.plot(pdrs, np.abs(signals - g0), linestyle="-", marker=".")
                ax_left.set_xlabel("Drive Power (a.u.)")
                ax_left.set_ylabel("Signal (a.u.)")
                ax_left.set_title(f"Current Scan (Iteration {i + 1})")

                # Right plot: Historical scans
                ax_right.clear()
                ax_right.plot(pdrs, np.abs(signals_array - g0).T, linestyle="--")
                ax_right.set_xlabel("Drive Power (a.u.)")
                ax_right.set_ylabel("Signal (a.u.)")
                ax_right.set_title("Historical Scans")

                dh.update(fig)

                while time.time() - start_t < interval:
                    plt.pause(0.5)  # Pause to allow the plot to update
            plt.close(fig)
            clear_output(wait=True)

            overnight_signals = np.array(overnight_signals)

            g0 = np.mean(overnight_signals[:, 0])

            # plot overnight_signals in one plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(pdrs, np.abs(overnight_signals - g0).T, linestyle="--")
            ax.set_xlabel(r"$\bar n$")
            ax.set_ylabel("Signal difference")
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            print("Overnight scans interrupted by user.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            iters = iters[: len(overnight_signals)]

        return iters, pdrs, overnight_signals

    def analyze(
        self,
        result: Optional[MISTPowerDepOvernightResultType] = None,
        *,
        g0=None,
        e0=None,
        ac_coeff=None,
    ) -> None:
        if result is None:
            result = self.last_result

        _, pdrs, signals = result

        if g0 is None:
            g0 = np.mean(signals[:, 0])

        abs_diff = np.abs(signals - g0)

        if ac_coeff is None:
            xs = pdrs
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * pdrs**2
            xlabel = r"$\bar n$"

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.plot(xs, abs_diff.T)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))

        plt.tight_layout()
        plt.show()

    def save(
        self,
        filepath: str,
        result: Optional[MISTPowerDepOvernightResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/mist/pdr_overnight",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        iters, pdrs, overnight_signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
            y_info={"name": "Iteration", "unit": "None", "values": iters},
            z_info={"name": "Signal", "unit": "a.u.", "values": overnight_signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

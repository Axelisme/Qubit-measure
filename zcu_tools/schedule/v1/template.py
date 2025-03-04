from typing import Any, Callable, Dict

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from qick.averager_program import AveragerProgram, RAveragerProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow1D


def default_R_result2signals(result) -> ndarray:
    xs, avgi, avgq = result
    return xs, avgi[0][0] + 1j * avgq[0][0]  # type: ignore


def default_result2signals(result) -> ndarray:
    avgi, avgq = result
    return avgi[0][0] + 1j * avgq[0][0]  # type: ignore


def default_signals2real(signals) -> ndarray:
    return np.abs(signals)


def sweep1D_hard_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: type,
    *,
    init_xs: ndarray,
    init_signals: ndarray,
    progress: bool,
    instant_show: bool,
    xlabel: str,
    ylabel: str,
    result2signals: Callable = default_R_result2signals,
    signals2real: Callable = default_signals2real,
    **kwargs,
):
    xs = init_xs.copy()
    signals = init_signals.copy()
    if instant_show:
        viewer = InstantShow1D(xs, x_label=xlabel, y_label=ylabel)

        def callback(ir, *args):
            nonlocal signals
            if len(args) == 1:
                (sum_d,) = args
                avg_d = [d / (ir + 1) for d in sum_d]
                signals = result2signals(avg_d)
            else:  # if ret_std == True
                raise NotImplementedError("std not implemented")
            viewer.update_show(signals2real(signals))
    else:
        callback = None  # type: ignore

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    try:
        prog: RAveragerProgram = prog_cls(soccfg, cfg)
        result = prog.acquire(soc, progress=progress, callback=callback, **kwargs)
        xs, signals = result2signals(result)
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.update_show(signals2real(signals), xs)
            viewer.close_show()

    return xs, signals


def sweep1D_soft_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: type,
    *,
    xs: ndarray,
    init_signals: ndarray,
    progress: bool,
    instant_show: bool,
    updateCfg: Callable,
    xlabel: str,
    ylabel: str,
    result2signals: Callable = default_result2signals,
    signals2real: Callable = default_signals2real,
    **kwargs,
):
    signals = init_signals.copy()
    if instant_show:
        viewer = InstantShow1D(xs, x_label=xlabel, y_label=ylabel)
        show_period = int(len(xs) / 20 + 0.99)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    try:
        for i, x in enumerate(tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)):
            updateCfg(cfg, i, x)

            set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

            prog: AveragerProgram = prog_cls(soccfg, cfg)
            result = prog.acquire(soc, progress=False, **kwargs)
            signals[i] = result2signals(result)

            if instant_show and i % show_period == 0:
                viewer.update_show(signals2real(signals))

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.update_show(signals2real(signals))
            viewer.close_show()

    return xs, signals

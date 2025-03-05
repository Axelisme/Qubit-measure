from copy import deepcopy
from typing import Any, Callable, Dict, Tuple

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from zcu_tools.program.v2 import MyProgramV2
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow1D, InstantShow2D
from zcu_tools.tools import AsyncFunc, print_traceback


def default_result2signals(result) -> ndarray:
    return result[0][0].dot([1, 1j])


def default_signal2real(signals: ndarray) -> ndarray:
    return np.abs(signals)


def raw2result(ir, *args):
    if len(args) == 1:
        (sum_d,) = args
        avg_d = [d / (ir + 1) for d in sum_d]
        return avg_d

    # if ret_std == True
    sum_d, sum2_d = args
    avg_d = [d / (ir + 1) for d in sum_d]
    std_d = [np.sqrt(d2 / (ir + 1) - d**2) for d, d2 in zip(avg_d, sum2_d)]
    return avg_d, std_d


def sweep_hard_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: type,
    *,
    init_signals: ndarray,
    ticks: Tuple[ndarray, ...],
    xlabel: str,
    ylabel: str,
    result2signals: Callable = default_result2signals,
    signals2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[MyProgramV2, ndarray]:
    signals = init_signals.copy()

    ViewerCls = {1: InstantShow1D, 2: InstantShow2D}[len(ticks)]

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    with ViewerCls(*ticks, xlabel, ylabel) as viewer:

        def callback(ir, *args):
            nonlocal signals
            signals = result2signals(raw2result(ir, *args))
            viewer.update_show(signals2real(signals))

        try:
            prog: MyProgramV2 = prog_cls(soccfg, cfg)

            result = prog.acquire(soc, progress=progress, callback=callback, **kwargs)
            signals: ndarray = result2signals(result)
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signals2real(signals))

    return prog, signals


def sweep1D_soft_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: type,
    *,
    xs: ndarray,
    init_signals: ndarray,
    updateCfg: Callable,
    xlabel: str,
    ylabel: str,
    result2signals: Callable = default_result2signals,
    signals2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals = init_signals.copy()

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    with InstantShow1D(xs, xlabel, ylabel) as viewer:
        try:
            with AsyncFunc(viewer.update_show, include_idx=False) as async_draw:
                for i, x in enumerate(
                    tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)
                ):
                    updateCfg(cfg, i, x)

                    # set again in case of change
                    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                    prog: MyProgramV2 = prog_cls(soccfg, cfg)
                    result = prog.acquire(soc, progress=False, **kwargs)
                    signals[i] = result2signals(result)

                    async_draw(i, signals2real(signals))

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signals2real(signals))

    return xs, signals


def sweep2D_soft_hard_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: type,
    *,
    xs: ndarray,
    ys: ndarray,
    init_signals: ndarray,
    updateCfg: Callable,
    xlabel: str,
    ylabel: str,
    result2signals: Callable = default_result2signals,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals2D = init_signals.copy()

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])  # set initial flux

    with InstantShow2D(xs, ys, xlabel, ylabel) as viewer:
        try:
            soft_tqdm = tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)
            avgs_tqdm = tqdm(total=cfg["soft_avgs"], smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update_show, include_idx=False) as async_draw:
                for i, x in enumerate(soft_tqdm):
                    updateCfg(cfg, i, x)

                    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                    avgs_tqdm.reset()
                    avgs_tqdm.refresh()

                    _signals2D = signals2D.copy()  # prevent overwrite

                    def callback(ir, *args):
                        nonlocal _signals2D
                        avgs_tqdm.update(max(ir + 1 - avgs_tqdm.n, 0))
                        avgs_tqdm.refresh()

                        _signals2D[i] = result2signals(raw2result(ir, *args))
                        viewer.update_show(signal2real(_signals2D))

                    prog = prog_cls(soccfg, cfg)
                    result = prog.acquire(
                        soc, progress=False, callback=callback, **kwargs
                    )
                    signals2D[i] = result2signals(result)

                    avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
                    avgs_tqdm.refresh()

                    async_draw(i, signal2real(signals2D))

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signal2real(signals2D), (xs, ys))
            soft_tqdm.close()
            avgs_tqdm.close()

    return prog, signals2D

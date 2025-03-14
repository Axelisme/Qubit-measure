from copy import deepcopy
from typing import Any, Callable, Dict, Tuple, Type

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from zcu_tools.program.v2 import MyProgramV2
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow1D, InstantShow2D
from zcu_tools.tools import AsyncFunc, print_traceback


def default_result2signals(*result) -> ndarray:
    avg_d = result[0][0][0].dot([1, 1j])
    if len(result) == 1:
        return avg_d
    else:
        std_d = np.abs(result[1][0][0].dot([1, 1j]))
        return avg_d, std_d


def default_signal2real(signals: ndarray) -> ndarray:
    return np.abs(signals)


def raw2result(ir, sum_d, sum2_d):
    avg_d = [d / (ir + 1) for d in sum_d]
    std_d = [np.sqrt(d2 / (ir + 1) - d**2) for d, d2 in zip(avg_d, sum2_d)]
    return avg_d, std_d


def sweep_hard_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: Type[MyProgramV2],
    *,
    ticks: Tuple[ndarray, ...],
    xlabel: str,
    ylabel: str,
    result2signals: Callable = default_result2signals,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[MyProgramV2, ndarray]:
    signals = np.full(tuple(len(t) for t in ticks), np.nan, dtype=complex)

    ViewerCls = [InstantShow1D, InstantShow2D][len(ticks) - 1]

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    with ViewerCls(*ticks, xlabel, ylabel) as viewer:

        def callback(*raw):
            nonlocal signals
            print(raw)
            signals, stds = result2signals(raw2result(*raw))
            viewer.update_show(signal2real(signals), stds=stds)

        try:
            prog = prog_cls(soccfg, cfg)

            result = prog.acquire(soc, progress=progress, callback=callback, **kwargs)
            signals: ndarray = result2signals(result)
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signal2real(signals))

    return prog, signals


def sweep1D_soft_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: Type[MyProgramV2],
    *,
    xs: ndarray,
    updateCfg: Callable,
    xlabel: str,
    ylabel: str,
    result2signals: Callable = default_result2signals,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals = np.full_like(xs, np.nan, dtype=complex)

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    with InstantShow1D(xs, xlabel, ylabel) as viewer:
        try:
            xs_tqdm = tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update_show, include_idx=False) as async_draw:
                for i, x in enumerate(xs_tqdm):
                    updateCfg(cfg, i, x)

                    # set again in case of change
                    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                    prog = prog_cls(soccfg, cfg)
                    result = prog.acquire(soc, progress=False, **kwargs)
                    signals[i] = result2signals(result)

                    async_draw(i, signal2real(signals))

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signal2real(signals))

    return xs, signals


def sweep2D_soft_hard_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: Type[MyProgramV2],
    *,
    xs: ndarray,
    ys: ndarray,
    updateCfg: Callable,
    xlabel: str,
    ylabel: str,
    result2signals: Callable = default_result2signals,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])  # set initial flux

    with InstantShow2D(xs, ys, xlabel, ylabel) as viewer:
        try:
            xs_tqdm = tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)
            avgs_tqdm = tqdm(total=cfg["soft_avgs"], smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update_show, include_idx=False) as async_draw:
                for i, x in enumerate(xs_tqdm):
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
            xs_tqdm.close()
            avgs_tqdm.close()

    return prog, signals2D

from copy import deepcopy
from typing import Any, Callable, Dict, Tuple, Type

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from zcu_tools.program.v1 import (
    MyAveragerProgram,
    MyNDAveragerProgram,
    MyRAveragerProgram,
)
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow1D, InstantShow2D
from zcu_tools.tools import AsyncFunc, print_traceback


def default_result2signals(result) -> ndarray:
    avgi, avgq, *_ = result
    return avgi[0][0] + 1j * avgq[0][0]  # type: ignore


def default_signal2real(signals) -> ndarray:
    return np.abs(signals)


def raw2result(ir, *args):
    if len(args) == 1:
        (sum_d,) = args
    else:
        sum_d, sum2_d = args

    sum_d = [d.dot([1, 1j]) for d in sum_d]
    avg_d = [d / (ir + 1) for d in sum_d]
    avgi_d = [d.real for d in avg_d]
    avgq_d = [d.imag for d in avg_d]

    if len(args) == 1:
        return avgi_d, avgq_d
    else:
        sum2_d = [d.dot([1, 1j]) for d in sum2_d]
        std_d = [np.sqrt(d2 / (ir + 1) - d**2) for d, d2 in zip(avg_d, sum2_d)]
        stdi_d = [d.real for d in std_d]
        stdq_d = [d.imag for d in std_d]

        return avgi_d, avgq_d, stdi_d, stdq_d


def sweep1D_hard_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: Type[MyRAveragerProgram],
    *,
    xs: ndarray,
    xlabel: str,
    ylabel: str,
    result2signals: Callable = default_result2signals,
    signals2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[ndarray, ndarray]:
    signals = np.full_like(xs, np.nan, dtype=complex)

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    with InstantShow1D(xs, xlabel, ylabel) as viewer:

        def callback(ir, *args):
            nonlocal signals
            signals = result2signals(raw2result(ir, *args))
            viewer.update_show(signals2real(signals))

        try:
            prog = prog_cls(soccfg, cfg)

            xs, *result = prog.acquire(
                soc, progress=progress, callback=callback, **kwargs
            )
            signals: ndarray = result2signals(result)
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signals2real(signals))

    return xs, signals


def sweep1D_soft_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: Type[MyAveragerProgram],
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


def sweep2D_hard_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: Type[MyNDAveragerProgram],
    *,
    xs: ndarray,
    ys: ndarray,
    xlabel: str,
    ylabel: str,
    result2signals: Callable = default_result2signals,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[ndarray, ndarray, ndarray]:
    signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])  # set initial flux

    with InstantShow2D(xs, ys, xlabel, ylabel) as viewer:

        def callback(ir, *args):
            nonlocal signals2D
            signals2D = result2signals(raw2result(ir, *args))
            viewer.update_show(signal2real(signals2D))

        try:
            prog = prog_cls(soccfg, cfg)
            xs_ys, *result = prog.acquire(
                soc, progress=progress, callback=callback, **kwargs
            )
            signals2D = result2signals(result)
            xs, ys = xs_ys[0], xs_ys[1]
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signal2real(signals2D), (xs, ys))

    return xs, ys, signals2D


def sweep2D_soft_hard_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: Type[MyRAveragerProgram],
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
                    xs, *result = prog.acquire(
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

    return xs, ys, signals2D


def sweep2D_soft_soft_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: Type[MyAveragerProgram],
    *,
    xs: ndarray,
    ys: ndarray,
    x_updateCfg: Callable,
    y_updateCfg: Callable,
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
            ys_tqdm = tqdm(ys, desc=ylabel, smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update_show, include_idx=False) as async_draw:
                for i, x in enumerate(xs):
                    x_updateCfg(cfg, i, x)

                    ys_tqdm.reset()
                    ys_tqdm.refresh()
                    for j, y in enumerate(ys):
                        y_updateCfg(cfg, j, y)

                        set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                        prog = prog_cls(soccfg, cfg)
                        result = prog.acquire(soc, progress=False, **kwargs)
                        signals2D[i, j] = result2signals(result)

                        ys_tqdm.update()

                        async_draw(i * len(ys) + j, signal2real(signals2D))
                    xs_tqdm.update()

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signal2real(signals2D), (xs, ys))
            xs_tqdm.close()
            ys_tqdm.close()

    return xs, ys, signals2D

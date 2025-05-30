from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from zcu_tools.program.v1 import (
    MyAveragerProgram,
    MyNDAveragerProgram,
    MyRAveragerProgram,
)
from zcu_tools.tools import AsyncFunc, print_traceback

from ..flux import set_flux
from ..instant_show import (
    InstantShow1D,
    InstantShow2D,
    InstantShowScatter,
)


def default_result2signals(
    avgi: list, avgq: list, stdi: list, stdq: list
) -> Tuple[ndarray, Optional[ndarray]]:
    signals = avgi[0][0] + 1j * avgq[0][0]  # (*sweep)
    stds = np.maximum(stdi[0][0], stdq[0][0])  # (*sweep)
    return signals, stds


def default_signal2real(signals) -> ndarray:
    return np.abs(signals)


def std2err(stds: Optional[ndarray], N: int) -> Optional[ndarray]:
    if stds is None:
        return None
    return stds / np.sqrt(N)


def raw2result(ir, sum_d, sum2_d):
    avg_d = [d / (ir + 1) for d in sum_d]
    avgi = [d[..., 0] for d in avg_d]
    avgq = [d[..., 1] for d in avg_d]

    std_d = [np.sqrt(s2 / (ir + 1) - u**2) for u, s2 in zip(avg_d, sum2_d)]
    stdi = [d[..., 0] for d in std_d]
    stdq = [d[..., 1] for d in std_d]

    return avgi, avgq, stdi, stdq


def sweep1D_hard_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_cls: Type[MyRAveragerProgram],
    *,
    xs: ndarray,
    xlabel: str,
    ylabel: str,
    result2signals: Callable[
        [list, list, list, list], Tuple[ndarray, Optional[ndarray]]
    ] = default_result2signals,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[ndarray, ndarray]:
    signals = np.full_like(xs, np.nan, dtype=complex)

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    title = None
    with InstantShow1D(xs, xlabel, ylabel, title=title) as viewer:
        try:
            prog = prog_cls(soccfg, cfg)

            def callback(ir, sum_d, sum2_d):
                nonlocal signals, title
                signals, _ = result2signals(*raw2result(ir, sum_d, sum2_d))
                viewer.update_show(
                    signal2real(signals),
                    title=title,
                )

            xs, *result = prog.acquire(
                soc, progress=progress, callback=callback, **kwargs
            )
            signals, _ = result2signals(*result)
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(
                signal2real(signals),
                ticks=xs,
                title=title,
            )

    return xs, signals


def sweep1D_soft_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_or_fn: Union[Type[MyAveragerProgram], Callable],
    *,
    xs: ndarray,
    updateCfg: Callable,
    xlabel: str,
    ylabel: str,
    result2signals: Callable[
        [list, list, list, list], Tuple[ndarray, Optional[ndarray]]
    ] = default_result2signals,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals = np.full_like(xs, np.nan, dtype=complex)
    stds = np.full_like(xs, np.nan, dtype=float)

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    with InstantShow1D(xs, xlabel, ylabel) as viewer:
        try:
            xs_tqdm = tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update_show, include_idx=False) as async_draw:
                for i, x in enumerate(xs_tqdm):
                    updateCfg(cfg, i, x)

                    # set again in case of change
                    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                    if isinstance(prog_or_fn, type):
                        prog = prog_or_fn(soccfg, cfg)
                        avgi, avgq, stdi, stdq = prog.acquire(
                            soc, progress=False, **kwargs
                        )
                        signals[i], stds[i] = result2signals(avgi, avgq, stdi, stdq)
                    elif isinstance(prog_or_fn, Callable):
                        signals[i], stds[i] = prog_or_fn(soc, soccfg, cfg)
                    else:
                        raise ValueError("prog_or_fn must be a type or a callable")

                    async_draw(i, signal2real(signals))

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signal2real(signals))
            xs_tqdm.close()

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
    result2signals: Callable[
        [list, list, list, list], Tuple[ndarray, Optional[ndarray]]
    ] = default_result2signals,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[ndarray, ndarray, ndarray]:
    signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    # set initial flux
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    with InstantShow2D(xs, ys, xlabel, ylabel) as viewer:

        def callback(ir, sum_d, sum2_d):
            nonlocal signals2D
            signals2D, _ = result2signals(*raw2result(ir, sum_d, sum2_d))
            viewer.update_show(signal2real(signals2D))

        try:
            prog = prog_cls(soccfg, cfg)
            xs_ys, *result = prog.acquire(
                soc, progress=progress, callback=callback, **kwargs
            )
            signals2D, _ = result2signals(*result)
            xs, ys = xs_ys[0], xs_ys[1]
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signal2real(signals2D), ticks=(xs, ys))

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
    result2signals: Callable[
        [list, list, list, list], Tuple[ndarray, Optional[ndarray]]
    ] = default_result2signals,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    early_stop_checker: Optional[Callable[[ndarray], Tuple[bool, str]]] = None,
    **kwargs,
) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    # set initial flux
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    title = None
    with InstantShow2D(xs, ys, xlabel, ylabel, title=title) as viewer:
        try:
            xs_tqdm = tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)
            avgs_tqdm = tqdm(total=cfg["soft_avgs"], smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update_show, include_idx=False) as async_draw:
                for i, x in enumerate(xs_tqdm):
                    updateCfg(cfg, i, x)

                    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                    avgs_tqdm.total = cfg["soft_avgs"]
                    avgs_tqdm.reset()
                    avgs_tqdm.refresh()

                    prog = prog_cls(soccfg, cfg)

                    _signals2D = signals2D.copy()  # prevent overwrite

                    def callback(ir, sum_d, sum2_d):
                        nonlocal _signals2D, avgs_tqdm, title
                        avgs_tqdm.update(max(ir + 1 - avgs_tqdm.n, 0))
                        avgs_tqdm.refresh()

                        _signals2D[i], _ = result2signals(
                            *raw2result(ir, sum_d, sum2_d)
                        )
                        if early_stop_checker is not None:
                            stop, title = early_stop_checker(_signals2D[i])
                            if stop:
                                prog.set_early_stop()

                        viewer.update_show(signal2real(_signals2D), title=title)

                    xs, *result = prog.acquire(
                        soc, progress=False, callback=callback, **kwargs
                    )
                    signals2D[i], _ = result2signals(*result)

                    avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
                    avgs_tqdm.refresh()

                    async_draw(i, signal2real(signals2D), ticks=(xs, ys), title=title)

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signal2real(signals2D), ticks=(xs, ys), title=title)
            xs_tqdm.close()
            avgs_tqdm.close()

    return xs, ys, signals2D


def sweep2D_soft_soft_template(
    soc,
    soccfg,
    cfg: Dict[str, Any],
    prog_or_fn: Union[Type[MyAveragerProgram], Callable],
    *,
    xs: ndarray,
    ys: ndarray,
    x_updateCfg: Callable,
    y_updateCfg: Callable,
    xlabel: str,
    ylabel: str,
    result2signals: Callable[
        [list, list, list, list], Tuple[ndarray, Optional[ndarray]]
    ] = default_result2signals,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    # set initial flux
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

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

                        if isinstance(prog_or_fn, type):
                            prog = prog_or_fn(soccfg, cfg)
                            result = prog.acquire(soc, progress=False, **kwargs)
                            signals2D[i, j], _ = result2signals(*result)
                        elif isinstance(prog_or_fn, Callable):
                            signals2D[i, j], _ = prog_or_fn(soc, soccfg, cfg)
                        else:
                            raise ValueError("prog_or_fn must be a type or a callable")

                        ys_tqdm.update()

                        async_draw(i * len(ys) + j, signal2real(signals2D))
                    xs_tqdm.update()

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signal2real(signals2D), ticks=(xs, ys))
            xs_tqdm.close()
            ys_tqdm.close()

    return xs, ys, signals2D


def sweep2D_maximize_template(
    measure_fn: Callable,
    *,
    xs: ndarray,
    ys: ndarray,
    xlabel: str,
    ylabel: str,
    signals2score: Callable,
    method: str = "Nelder-Mead",
    **kwargs,
):
    from scipy.optimize import minimize

    with InstantShowScatter(xlabel, ylabel) as viewer:

        def loss_func(param):
            x, y = param

            signals, _ = measure_fn(x, y)
            score = signals2score(signals)

            viewer.append_spot(x, y, score)

            return -score

        options = dict(maxiter=(len(xs) * len(ys)) // 5)

        if method in ["Nelder-Mead", "Powell"]:
            options["xatol"] = min(xs[1] - xs[0], ys[1] - ys[0])
        elif method in ["L-BFGS-B"]:
            options["ftol"] = 1e-4  # type: ignore

        options.update(kwargs)

        init_point = (0.5 * (xs[0] + xs[-1]), 0.5 * (ys[0] + ys[-1]))
        res = minimize(
            loss_func,
            init_point,
            method=method,
            bounds=[(xs[0], xs[-1]), (ys[0], ys[-1])],
            options=options,
        )

    if isinstance(res, ndarray):
        return res
    return res.x

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from zcu_tools.liveplot.jupyter import (
    LivePlotter1D,
    LivePlotter2D,
    LivePlotter2DwithLine,
)
from zcu_tools.program.v2 import MyProgramV2
from zcu_tools.tools import AsyncFunc, print_traceback

from ..flux import set_flux


def default_result2signal(
    avg_d: list, std_d: list
) -> Tuple[ndarray, Optional[ndarray]]:
    avg_d = avg_d[0][0].dot([1, 1j])  # (*sweep)
    std_d = np.max(std_d[0][0], axis=-1)  # (*sweep)

    return avg_d, std_d


def default_signal2real(signals: ndarray) -> ndarray:
    return np.abs(signals)


def std2err(stds: Optional[ndarray], N: int) -> Optional[ndarray]:
    if stds is None:
        return None
    return stds / np.sqrt(N)


def raw2result(ir, sum_d, sum2_d) -> Tuple[ndarray, ndarray]:
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
    result2signals: Callable[
        [list, list], Tuple[ndarray, Optional[ndarray]]
    ] = default_result2signal,
    signal2real: Callable[[ndarray], ndarray] = default_signal2real,
    progress: bool = True,
    viewer_kwargs: Dict[str, Any] = {},
) -> Tuple[MyProgramV2, ndarray]:
    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    ViewerCls = [LivePlotter1D, LivePlotter2D][len(ticks) - 1]

    prog = None
    title = None
    signals = np.full(tuple(len(t) for t in ticks), np.nan, dtype=complex)
    with ViewerCls(xlabel, ylabel, title=title, **viewer_kwargs) as viewer:

        def callback(ir, sum_d, sum2_d) -> None:
            nonlocal signals, title
            signals, _ = result2signals(*raw2result(ir, sum_d, sum2_d))
            viewer.update(*ticks, signal2real(signals), title=title)

        try:
            prog = prog_cls(soccfg, cfg)

            results = prog.acquire(soc, progress=progress, callback=callback)
            signals, _ = result2signals(*results)
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception as e:
            if prog is None:
                raise e  # the error is happen in initialize of program
            print("Error during measurement:")
            print_traceback()
        viewer.update(*ticks, signal2real(signals), title=title)

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
    result2signals: Callable[
        [list, list], Tuple[ndarray, Optional[ndarray]]
    ] = default_result2signal,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals = np.full_like(xs, np.nan, dtype=complex)
    stds = np.full_like(xs, np.nan, dtype=float)

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    with LivePlotter1D(xlabel, ylabel) as viewer:
        try:
            xs_tqdm = tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update, include_idx=False) as async_draw:
                for i, x in enumerate(xs_tqdm):
                    updateCfg(cfg, i, x)

                    # set again in case of change
                    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                    prog = prog_cls(soccfg, cfg)
                    result = prog.acquire(soc, progress=False)
                    signals[i], stds[i] = result2signals(*result)

                    async_draw(i, xs, signal2real(signals))

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        viewer.update(xs, signal2real(signals))

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
    result2signals: Callable[
        [list, list], Tuple[ndarray, Optional[ndarray]]
    ] = default_result2signal,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    num_lines: int = 1,
) -> Tuple[MyProgramV2, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    # set initial flux
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    prog = None
    title = None
    with LivePlotter2DwithLine(
        xlabel, ylabel, line_axis=1, num_lines=num_lines, title=title
    ) as viewer:
        try:
            xs_tqdm = tqdm(xs, desc=xlabel, smoothing=0, disable=not progress)
            avgs_tqdm = tqdm(total=cfg["soft_avgs"], smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update, include_idx=False) as async_draw:
                for i, x in enumerate(xs_tqdm):
                    updateCfg(cfg, i, x)

                    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                    avgs_tqdm.total = cfg["soft_avgs"]
                    avgs_tqdm.reset()
                    avgs_tqdm.refresh()

                    _signals2D = signals2D.copy()  # prevent overwrite

                    def callback(ir, sum_d, sum2_d) -> None:
                        nonlocal _signals2D, title, avgs_tqdm
                        avgs_tqdm.update(max(ir + 1 - avgs_tqdm.n, 0))
                        avgs_tqdm.refresh()

                        _signals2D[i], _ = result2signals(
                            *raw2result(ir, sum_d, sum2_d)
                        )
                        signals_real = signal2real(_signals2D)
                        viewer.update(xs, ys, signals_real, i, title=title)

                    prog = prog_cls(soccfg, cfg)

                    avg_d, std_d = prog.acquire(soc, progress=False, callback=callback)
                    signals2D[i], _ = result2signals(avg_d, std_d)

                    avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
                    avgs_tqdm.refresh()

                    async_draw(i, xs, ys, signal2real(signals2D), i, title=title)

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
            viewer.update(xs, ys, signal2real(signals2D), i, title=title)
        except Exception as e:
            if prog is None:
                raise e  # the error is happen in initialize of program
            print("Error during measurement:")
            print_traceback()
        finally:
            xs_tqdm.close()
            avgs_tqdm.close()

    return prog, signals2D

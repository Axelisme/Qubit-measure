from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from zcu_tools.program.v2 import MyProgramV2
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow1D, InstantShow2D
from zcu_tools.tools import AsyncFunc, print_traceback


def default_result2signals(
    avg_d: list, std_d: list
) -> Tuple[ndarray, Optional[ndarray]]:
    avg_d = avg_d[0][0].dot([1, 1j])
    std_d = np.max(std_d[0][0], axis=-1)

    return avg_d, std_d


def default_signal2real(signals: ndarray) -> ndarray:
    return np.abs(signals)


def std2err(stds: Optional[ndarray], N: int) -> Optional[ndarray]:
    if stds is None:
        return None
    return stds / np.sqrt(N)


def raw2result(ir, sum_d, sum2_d) -> Tuple[np.ndarray, np.ndarray]:
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
    ] = default_result2signals,
    signal2real: Callable[[ndarray], ndarray] = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[MyProgramV2, ndarray]:
    signals = np.full(tuple(len(t) for t in ticks), np.nan, dtype=complex)
    stds = np.full_like(signals, np.nan, dtype=float)

    reps = cfg["reps"]

    ViewerCls = [InstantShow1D, InstantShow2D][len(ticks) - 1]

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog = None
    with ViewerCls(*ticks, xlabel, ylabel) as viewer:

        def callback(ir, sum_d, sum2_d):
            nonlocal signals, stds
            signals, stds = result2signals(*raw2result(ir, sum_d, sum2_d))
            viewer.update_show(
                signal2real(signals), errs=std2err(stds, (ir + 1) * reps)
            )

        try:
            prog = prog_cls(soccfg, cfg)

            avg_d, std_d = prog.acquire(
                soc, progress=progress, callback=callback, **kwargs
            )
            signals, stds = result2signals(avg_d, std_d)
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(
                signal2real(signals), errs=std2err(stds, cfg["soft_avgs"] * reps)
            )

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
    ] = default_result2signals,
    signal2real: Callable = default_signal2real,
    progress: bool = True,
    **kwargs,
) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals = np.full_like(xs, np.nan, dtype=complex)
    stds = np.full_like(xs, np.nan, dtype=float)

    N = cfg["soft_avgs"] * cfg["reps"]

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
                    avg_d, std_d = prog.acquire(soc, progress=False, **kwargs)
                    signals[i], stds[i] = result2signals(avg_d, std_d)

                    async_draw(i, signal2real(signals), errs=std2err(stds, N))

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signal2real(signals), errs=std2err(stds, N))

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
    ] = default_result2signals,
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

                    def callback(ir, sum_d, sum2_d):
                        nonlocal _signals2D
                        avgs_tqdm.update(max(ir + 1 - avgs_tqdm.n, 0))
                        avgs_tqdm.refresh()

                        _signals2D[i], _ = result2signals(
                            *raw2result(ir, sum_d, sum2_d)
                        )
                        viewer.update_show(signal2real(_signals2D))

                    prog = prog_cls(soccfg, cfg)
                    avg_d, std_d = prog.acquire(
                        soc, progress=False, callback=callback, **kwargs
                    )
                    signals2D[i], _ = result2signals(avg_d, std_d)

                    avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
                    avgs_tqdm.refresh()

                    async_draw(i, signal2real(signals2D))

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update_show(signal2real(signals2D), ticks=(xs, ys))
            xs_tqdm.close()
            avgs_tqdm.close()

    return prog, signals2D

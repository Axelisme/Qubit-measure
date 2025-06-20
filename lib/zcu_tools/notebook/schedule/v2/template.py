from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from zcu_tools.liveplot import AbsLivePlotter
from zcu_tools.tools import AsyncFunc, print_traceback

from ..flux import set_flux


def raw2result(ir, sum_d, sum2_d) -> Tuple[ndarray, ndarray]:
    avg_d = [d / (ir + 1) for d in sum_d]
    std_d = [np.sqrt(d2 / (ir + 1) - d**2) for d, d2 in zip(avg_d, sum2_d)]
    return avg_d, std_d


def default_result2signal(
    avg_d: list, std_d: list
) -> Tuple[ndarray, Optional[ndarray]]:
    avg_d = avg_d[0][0].dot([1, 1j])  # (*sweep)
    std_d = np.max(std_d[0][0], axis=-1)  # (*sweep)

    return avg_d, std_d


def default_signal2real(signals: ndarray) -> ndarray:
    return np.abs(signals)


MeasureFn = Callable[
    [Dict[str, Any], Optional[Callable[[int, list, list], None]]],
    Tuple[list, list],
]
Result2SignalFn = Callable[[list, list], Tuple[ndarray, Optional[ndarray]]]
Signal2RealFn = Callable[[ndarray], ndarray]
UpdateCfgFn = Callable[[Dict[str, Any], int, Any], None]


def sweep_hard_template(
    cfg: Dict[str, Any],
    measure_fn: MeasureFn,
    liveplotter: AbsLivePlotter,
    *,
    ticks: Tuple[ndarray, ...],
    result2signals: Result2SignalFn = default_result2signal,
    signal2real: Signal2RealFn = default_signal2real,
    catch_interrupt: bool = True,
) -> ndarray:
    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    signals = np.full(tuple(len(t) for t in ticks), np.nan, dtype=complex)
    with liveplotter as viewer:

        def callback(ir, sum_d, sum2_d) -> None:
            nonlocal signals
            signals, _ = result2signals(*raw2result(ir, sum_d, sum2_d))
            viewer.update(*ticks, signal2real(signals))

        try:
            results = measure_fn(cfg, callback)
            signals, _ = result2signals(*results)
        except KeyboardInterrupt as e:
            if not catch_interrupt:
                raise e  # re-raise if not catching
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception as e:
            if not catch_interrupt:
                raise e  #
            print("Error during measurement:")
            print_traceback()
        viewer.update(*ticks, signal2real(signals))

    return signals


def sweep1D_soft_template(
    cfg: Dict[str, Any],
    measure_fn: MeasureFn,
    liveplotter: AbsLivePlotter,
    *,
    xs: ndarray,
    updateCfg: UpdateCfgFn,
    result2signals: Result2SignalFn = default_result2signal,
    signal2real: Signal2RealFn = default_signal2real,
    progress: bool = True,
    catch_interrupt: bool = True,
) -> ndarray:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals = np.full_like(xs, np.nan, dtype=complex)

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    with liveplotter as viewer:
        try:
            xs_tqdm = tqdm(xs, smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update, include_idx=False) as async_draw:
                for i, x in enumerate(xs_tqdm):
                    updateCfg(cfg, i, x)

                    # set again in case of change
                    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                    result = measure_fn(cfg, callback=None)
                    signals[i], _ = result2signals(*result)

                    async_draw(i, xs, signal2real(signals))

        except KeyboardInterrupt as e:
            if not catch_interrupt:
                raise e  #
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception as e:
            if not catch_interrupt:
                raise e  #
            print("Error during measurement:")
            print_traceback()
        viewer.update(xs, signal2real(signals))

    return signals


def sweep2D_soft_hard_template(
    cfg: Dict[str, Any],
    measure_fn: MeasureFn,
    liveplotter: AbsLivePlotter,
    *,
    xs: ndarray,
    ys: ndarray,
    updateCfg: UpdateCfgFn,
    result2signals: Result2SignalFn = default_result2signal,
    signal2real: Signal2RealFn = default_signal2real,
    progress: bool = True,
    catch_interrupt: bool = True,
) -> ndarray:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    # set initial flux
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    with liveplotter as viewer:
        try:
            xs_tqdm = tqdm(xs, smoothing=0, disable=not progress)
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
                        nonlocal _signals2D, avgs_tqdm
                        avgs_tqdm.update(max(ir + 1 - avgs_tqdm.n, 0))
                        avgs_tqdm.refresh()

                        _signals2D[i], _ = result2signals(
                            *raw2result(ir, sum_d, sum2_d)
                        )
                        signals_real = signal2real(_signals2D)
                        viewer.update(xs, ys, signals_real)

                    results = measure_fn(cfg, callback=callback)
                    signals2D[i], _ = result2signals(*results)

                    avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
                    avgs_tqdm.refresh()

                    async_draw(i, xs, ys, signal2real(signals2D))

        except KeyboardInterrupt as e:
            if not catch_interrupt:
                raise e  #
            print("Received KeyboardInterrupt, early stopping the program")
            viewer.update(xs, ys, signal2real(signals2D))
        except Exception as e:
            if not catch_interrupt:
                raise e  #
            print("Error during measurement:")
            print_traceback()
        finally:
            xs_tqdm.close()
            avgs_tqdm.close()

    return signals2D

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from zcu_tools.liveplot import AbsLivePlotter
from zcu_tools.tools import AsyncFunc, print_traceback

from ..flux import set_flux


def raw2result(ir: int, sum_d: list, sum2_d: list) -> Tuple[list, list, list, list]:
    avg_d = [d / (ir + 1) for d in sum_d]
    avgi = [d[..., 0] for d in avg_d]
    avgq = [d[..., 1] for d in avg_d]

    std_d = [np.sqrt(s2 / (ir + 1) - u**2) for u, s2 in zip(avg_d, sum2_d)]
    stdi = [d[..., 0] for d in std_d]
    stdq = [d[..., 1] for d in std_d]

    return avgi, avgq, stdi, stdq


def default_result2signals(
    avgi: list, avgq: list, stdi: list, stdq: list
) -> Tuple[ndarray, Optional[ndarray]]:
    signals = avgi[0][0] + 1j * avgq[0][0]  # (*sweep)
    stds = np.sqrt(stdi[0][0] ** 2 + stdq[0][0] ** 2)
    return signals, stds


def default_signal2real(signals: ndarray) -> ndarray:
    return np.abs(signals)


MeasureFn = Callable[[Dict[str, Any], Optional[Callable[[int, list, list], None]]], Any]
Result2SignalFn = Callable[[list, list], Tuple[ndarray, Optional[ndarray]]]
Signal2RealFn = Callable[[ndarray], ndarray]
UpdateCfgFn = Callable[[Dict[str, Any], int, Any], None]


def sweep1D_hard_template(
    cfg: Dict[str, Any],
    measure_fn: MeasureFn,
    liveplotter: AbsLivePlotter,
    *,
    xs: ndarray,
    result2signals: Result2SignalFn = default_result2signals,
    signal2real: Signal2RealFn = default_signal2real,
    catch_interrupt: bool = True,
) -> Tuple[ndarray, ndarray]:
    signals = np.full_like(xs, np.nan, dtype=complex)

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    with liveplotter as viewer:
        try:

            def callback(ir: int, sum_d: list, sum2_d: list) -> None:
                nonlocal signals, xs
                signals, _ = result2signals(*raw2result(ir, sum_d, sum2_d))
                viewer.update(xs, signal2real(signals))

            xs, *result = measure_fn(cfg, callback)
            signals, _ = result2signals(*result)
        except KeyboardInterrupt as e:
            if not catch_interrupt:
                raise e
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception as e:
            if not catch_interrupt:
                raise e
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update(xs, signal2real(signals))

    return xs, signals


def sweep1D_soft_template(
    cfg: Dict[str, Any],
    measure_fn: MeasureFn,
    liveplotter: AbsLivePlotter,
    *,
    xs: ndarray,
    updateCfg: UpdateCfgFn,
    result2signals: Result2SignalFn = default_result2signals,
    signal2real: Signal2RealFn = default_signal2real,
    progress: bool = True,
    catch_interrupt: bool = True,
) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals = np.full_like(xs, np.nan, dtype=complex)
    stds = np.full_like(xs, np.nan, dtype=float)

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    with liveplotter as viewer:
        try:
            xs_tqdm = tqdm(xs, smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update) as async_draw:
                assert async_draw is not None
                for i, x in enumerate(xs_tqdm):
                    updateCfg(cfg, i, x)

                    # set again in case of change
                    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                    results = measure_fn(cfg, callback=None)
                    signals[i], stds[i] = result2signals(*results)

                    async_draw(xs, signal2real(signals))

        except KeyboardInterrupt as e:
            if not catch_interrupt:
                raise e
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception as e:
            if not catch_interrupt:
                raise e
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update(xs, signal2real(signals))
            xs_tqdm.close()

    return xs, signals


def sweep2D_hard_template(
    cfg: Dict[str, Any],
    measure_fn: MeasureFn,
    liveplotter: AbsLivePlotter,
    *,
    xs: ndarray,
    ys: ndarray,
    result2signals: Result2SignalFn = default_result2signals,
    signal2real: Signal2RealFn = default_signal2real,
    catch_interrupt: bool = True,
) -> Tuple[ndarray, ndarray, ndarray]:
    signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    # set initial flux
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    with liveplotter as viewer:

        def callback(ir: int, sum_d: list, sum2_d: list) -> None:
            nonlocal signals2D, xs, ys
            signals2D, _ = result2signals(*raw2result(ir, sum_d, sum2_d))
            viewer.update(xs, ys, signal2real(signals2D))

        try:
            xs_ys, *result = measure_fn(cfg, callback)
            signals2D, _ = result2signals(*result)
            xs, ys = xs_ys[0], xs_ys[1]
        except KeyboardInterrupt as e:
            if not catch_interrupt:
                raise e
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception as e:
            if not catch_interrupt:
                raise e
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update(xs, ys, signal2real(signals2D))

    return xs, ys, signals2D


def sweep2D_soft_hard_template(
    cfg: Dict[str, Any],
    measure_fn: MeasureFn,
    liveplotter: AbsLivePlotter,
    *,
    xs: ndarray,
    ys: ndarray,
    updateCfg: UpdateCfgFn,
    result2signals: Result2SignalFn = default_result2signals,
    signal2real: Signal2RealFn = default_signal2real,
    progress: bool = True,
    catch_interrupt: bool = True,
) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    # set initial flux
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    with liveplotter as viewer:
        try:
            xs_tqdm = tqdm(xs, smoothing=0, disable=not progress)
            avgs_tqdm = tqdm(total=cfg["soft_avgs"], smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update) as async_draw:
                assert async_draw is not None
                for i, x in enumerate(xs_tqdm):
                    updateCfg(cfg, i, x)

                    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                    avgs_tqdm.total = cfg["soft_avgs"]
                    avgs_tqdm.reset()
                    avgs_tqdm.refresh()

                    _signals2D = signals2D.copy()  # prevent overwrite

                    def callback(ir: int, sum_d: list, sum2_d: list) -> None:
                        nonlocal _signals2D, avgs_tqdm
                        avgs_tqdm.update(max(ir + 1 - avgs_tqdm.n, 0))
                        avgs_tqdm.refresh()

                        _signals2D[i], _ = result2signals(
                            *raw2result(ir, sum_d, sum2_d)
                        )

                        viewer.update(xs, ys, signal2real(_signals2D))

                    xs, *results = measure_fn(cfg, callback)
                    signals2D[i], _ = result2signals(*results)

                    avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
                    avgs_tqdm.refresh()

                    async_draw(xs, ys, signal2real(signals2D))

        except KeyboardInterrupt as e:
            if not catch_interrupt:
                raise e
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception as e:
            if not catch_interrupt:
                raise e
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update(xs, ys, signal2real(signals2D))
            xs_tqdm.close()
            avgs_tqdm.close()

    return xs, ys, signals2D


def sweep2D_soft_soft_template(
    cfg: Dict[str, Any],
    measure_fn: MeasureFn,
    liveplotter: AbsLivePlotter,
    *,
    xs: ndarray,
    ys: ndarray,
    x_updateCfg: UpdateCfgFn,
    y_updateCfg: UpdateCfgFn,
    result2signals: Result2SignalFn = default_result2signals,
    signal2real: Signal2RealFn = default_signal2real,
    progress: bool = True,
    catch_interrupt: bool = True,
) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    # set initial flux
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    with liveplotter as viewer:
        try:
            xs_tqdm = tqdm(xs, smoothing=0, disable=not progress)
            ys_tqdm = tqdm(ys, smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update) as async_draw:
                assert async_draw is not None
                for i, x in enumerate(xs):
                    x_updateCfg(cfg, i, x)

                    ys_tqdm.reset()
                    ys_tqdm.refresh()
                    for j, y in enumerate(ys):
                        y_updateCfg(cfg, j, y)

                        set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

                        results = measure_fn(cfg, callback=None)
                        signals2D[i, j], _ = result2signals(*results)

                        ys_tqdm.update()

                        async_draw(xs, ys, signal2real(signals2D))
                    xs_tqdm.update()

        except KeyboardInterrupt as e:
            if not catch_interrupt:
                raise e
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception as e:
            if not catch_interrupt:
                raise e
            print("Error during measurement:")
            print_traceback()
        finally:
            viewer.update(xs, ys, signal2real(signals2D))
            xs_tqdm.close()
            ys_tqdm.close()

    return xs, ys, signals2D

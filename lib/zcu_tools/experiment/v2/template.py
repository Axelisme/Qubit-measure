from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.liveplot import AbsLivePlotter
from zcu_tools.utils.async_func import AsyncFunc
from zcu_tools.utils.debug import print_traceback

# TODO: support user controlled callback function

MeasureFn = Callable[[Dict[str, Any], Optional[Callable[..., None]]], ndarray]
Raw2SignalFn = Callable[[int, List[ndarray], Optional[List[ndarray]]], ndarray]
Signal2RealFn = Callable[[ndarray], ndarray]
UpdateCfgFn = Callable[[Dict[str, Any], int, Any], None]


def avg_as_signal(
    ir: int, avg_d: List[ndarray], std_d: Optional[List[ndarray]]
) -> ndarray:
    return avg_d[0][0].dot([1, 1j])  # (*sweep)


def take_signal_abs(signals: ndarray) -> ndarray:
    return np.abs(signals)


def sweep_hard_template(
    cfg: Dict[str, Any],
    measure_fn: MeasureFn,
    liveplotter: AbsLivePlotter,
    *,
    ticks: Tuple[ndarray, ...],
    raw2signal: Raw2SignalFn = avg_as_signal,
    signal2real: Signal2RealFn = take_signal_abs,
    catch_interrupt: bool = True,
) -> ndarray:
    # set flux first
    GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

    signals = np.full(tuple(len(t) for t in ticks), np.nan, dtype=complex)
    with liveplotter as viewer:

        def callback(
            ir: int, avg_d: List[ndarray], std_d: Optional[List[ndarray]]
        ) -> None:
            nonlocal signals
            signals = raw2signal(ir, avg_d, std_d)
            viewer.update(*ticks, signal2real(signals))

        try:
            signals = measure_fn(cfg, callback)
        except KeyboardInterrupt:
            if not catch_interrupt:
                raise
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            if not catch_interrupt:
                raise
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
    signal2real: Signal2RealFn = take_signal_abs,
    progress: bool = True,
    catch_interrupt: bool = True,
    data_shape: Optional[tuple] = None,
) -> ndarray:
    cfg = deepcopy(cfg)  # prevent in-place modification

    # set flux first
    GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

    if data_shape is not None:
        signals = np.full((len(xs), *data_shape), np.nan, dtype=complex)
    else:
        signals = np.full_like(xs, np.nan, dtype=complex)

    with liveplotter as viewer:
        try:
            xs_tqdm = tqdm(xs, smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update) as async_draw:
                assert async_draw is not None
                for i, x in enumerate(xs_tqdm):
                    updateCfg(cfg, i, x)

                    # set again in case of change
                    GlobalDeviceManager.setup_devices(cfg["dev"], progress=False)

                    signals[i] = measure_fn(cfg, None)

                    async_draw(xs, signal2real(signals))

        except KeyboardInterrupt:
            if not catch_interrupt:
                raise
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            if not catch_interrupt:
                raise
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
    raw2signal: Raw2SignalFn = avg_as_signal,
    signal2real: Signal2RealFn = take_signal_abs,
    progress: bool = True,
    catch_interrupt: bool = True,
    data_shape: Optional[tuple] = None,
) -> ndarray:
    cfg = deepcopy(cfg)  # prevent in-place modification

    # set initial flux
    GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

    if data_shape is not None:
        signals2D = np.full((len(xs), len(ys), *data_shape), np.nan, dtype=complex)
    else:
        signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    with liveplotter as viewer:
        try:
            xs_tqdm = tqdm(xs, smoothing=0, disable=not progress)
            avgs_tqdm = tqdm(total=cfg["rounds"], smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update) as async_draw:
                assert async_draw is not None
                for i, x in enumerate(xs_tqdm):
                    updateCfg(cfg, i, x)

                    GlobalDeviceManager.setup_devices(cfg["dev"], progress=False)

                    avgs_tqdm.total = cfg["rounds"]
                    avgs_tqdm.reset()
                    avgs_tqdm.refresh()

                    _signals2D = signals2D.copy()  # prevent overwrite

                    def callback(
                        ir: int, avg_d: List[ndarray], std_d: Optional[List[ndarray]]
                    ) -> None:
                        nonlocal _signals2D, avgs_tqdm
                        avgs_tqdm.update(max(ir - avgs_tqdm.n, 0))
                        avgs_tqdm.refresh()

                        _signals2D[i] = raw2signal(ir, avg_d, std_d)
                        viewer.update(xs, ys, signal2real(_signals2D))

                    signals2D[i] = measure_fn(cfg, callback)

                    avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
                    avgs_tqdm.refresh()

                    async_draw(xs, ys, signal2real(signals2D))

        except KeyboardInterrupt:
            if not catch_interrupt:
                raise
            print("Received KeyboardInterrupt, early stopping the program")
            viewer.update(xs, ys, signal2real(signals2D))
        except Exception:
            if not catch_interrupt:
                raise
            print("Error during measurement:")
            print_traceback()
        finally:
            xs_tqdm.close()
            avgs_tqdm.close()

    return signals2D


def sweep2D_soft_template(
    cfg: Dict[str, Any],
    measure_fn: MeasureFn,
    liveplotter: AbsLivePlotter,
    *,
    xs: ndarray,
    ys: ndarray,
    updateCfg_x: UpdateCfgFn,
    updateCfg_y: UpdateCfgFn,
    signal2real: Signal2RealFn = take_signal_abs,
    progress: bool = True,
    catch_interrupt: bool = True,
    data_shape: Optional[tuple] = None,
) -> ndarray:
    cfg = deepcopy(cfg)  # prevent in-place modification

    # set initial flux
    GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

    if data_shape is not None:
        signals2D = np.full((len(xs), len(ys), *data_shape), np.nan, dtype=complex)
    else:
        signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    with liveplotter as viewer:
        try:
            xs_tqdm = tqdm(xs, smoothing=0, disable=not progress)
            ys_tqdm = tqdm(ys, smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update) as async_draw:
                assert async_draw is not None
                for i, x in enumerate(xs_tqdm):
                    updateCfg_x(cfg, i, x)

                    ys_tqdm.reset()
                    ys_tqdm.refresh()

                    for j, y in enumerate(ys):
                        updateCfg_y(cfg, j, y)

                        GlobalDeviceManager.setup_devices(cfg["dev"], progress=False)

                        signals2D[i, j] = measure_fn(cfg, None)

                        ys_tqdm.update()
                        ys_tqdm.refresh()

                        async_draw(xs, ys, signal2real(signals2D))

        except KeyboardInterrupt:
            if not catch_interrupt:
                raise
            print("Received KeyboardInterrupt, early stopping the program")
            viewer.update(xs, ys, signal2real(signals2D))
        except Exception:
            if not catch_interrupt:
                raise
            print("Error during measurement:")
            print_traceback()
        finally:
            xs_tqdm.close()
            ys_tqdm.close()

    return signals2D

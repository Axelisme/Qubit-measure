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
CallbackFn = Callable[[int, List[ndarray]], None]
MeasureFn = Callable[[Dict[str, Any], Optional[CallbackFn]], ndarray]
Raw2SignalFn = Callable[[int, List[ndarray]], ndarray]
Signal2RealFn = Callable[[ndarray], ndarray]
UpdateCfgFn = Callable[[Dict[str, Any], int, Any], None]


def avg_as_signal(ir: int, avg_d: List[ndarray], *args, **kwargs) -> ndarray:
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
    init_signals: Optional[ndarray] = None,
    realsignal_callback: Optional[Callable[[int, np.ndarray], None]] = None,
) -> ndarray:
    # set flux first
    GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

    if init_signals is not None:
        signals = init_signals.copy()
    else:
        signals = np.full(tuple(len(t) for t in ticks), np.nan, dtype=complex)

    with liveplotter as viewer:

        def callback(ir: int, *args, **kwargs) -> None:
            nonlocal signals
            signals = raw2signal(ir, *args, **kwargs)
            real_signals = signal2real(signals)
            viewer.update(*ticks, real_signals)

            if realsignal_callback is not None:
                realsignal_callback(ir, real_signals)

        try:
            signals = measure_fn(cfg, callback)
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
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
    init_signals: Optional[ndarray] = None,
    updateCfg: UpdateCfgFn,
    signal2real: Signal2RealFn = take_signal_abs,
    progress: bool = True,
) -> ndarray:
    cfg = deepcopy(cfg)  # prevent in-place modification

    # set flux first
    GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

    if init_signals is not None:
        signals = init_signals.copy()
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
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
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
    ticks: Tuple[ndarray, ...],
    init_signals: Optional[ndarray] = None,
    updateCfg: UpdateCfgFn,
    raw2signal: Raw2SignalFn = avg_as_signal,
    signal2real: Signal2RealFn = take_signal_abs,
    progress: bool = True,
    realsignal_callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
) -> ndarray:
    cfg = deepcopy(cfg)  # prevent in-place modification

    # set initial flux
    GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

    if init_signals is not None:
        signals = init_signals.copy()
    else:
        signals = np.full(
            (len(xs), *tuple(len(t) for t in ticks)), np.nan, dtype=complex
        )

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

                    _signals = signals.copy()  # prevent overwrite

                    def callback(ir: int, *args, **kwargs) -> None:
                        nonlocal _signals, avgs_tqdm
                        avgs_tqdm.update(max(ir - avgs_tqdm.n, 0))
                        avgs_tqdm.refresh()

                        _signals[i] = raw2signal(ir, *args, **kwargs)
                        real_signals = signal2real(_signals)
                        viewer.update(xs, *ticks, real_signals)

                        if realsignal_callback is not None:
                            realsignal_callback(i, ir, real_signals)

                    signals[i] = measure_fn(cfg, callback)

                    avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
                    avgs_tqdm.refresh()

                    async_draw(xs, *ticks, signal2real(signals))

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
            viewer.update(xs, *ticks, signal2real(signals))
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            xs_tqdm.close()
            avgs_tqdm.close()

    return signals


def sweep2D_soft_template(
    cfg: Dict[str, Any],
    measure_fn: MeasureFn,
    liveplotter: AbsLivePlotter,
    *,
    xs: ndarray,
    ys: ndarray,
    init_signals: Optional[ndarray] = None,
    updateCfg_x: UpdateCfgFn,
    updateCfg_y: UpdateCfgFn,
    signal2real: Signal2RealFn = take_signal_abs,
    progress: bool = True,
) -> ndarray:
    cfg = deepcopy(cfg)  # prevent in-place modification

    # set initial flux
    GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

    if init_signals is not None:
        signals2D = init_signals.copy()
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
            print("Received KeyboardInterrupt, early stopping the program")
            viewer.update(xs, ys, signal2real(signals2D))
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            xs_tqdm.close()
            ys_tqdm.close()

    return signals2D


def sweep_soft_batch_template(
    cfg: Dict[str, Any],
    measure_blocks: Dict[
        str,
        Tuple[
            Callable[
                [Dict[str, Any], int, List[ndarray], Optional[CallbackFn]], ndarray
            ],
            Tuple[ndarray, ...],
            Optional[Raw2SignalFn],
            Optional[Signal2RealFn],
            Optional[Callable[[int, int, np.ndarray], None]],
        ],
    ],
    liveplotter: AbsLivePlotter,
    *,
    xs: ndarray,
    updateCfg: UpdateCfgFn,
    progress: bool = True,
) -> Dict[str, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    # set initial flux
    GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

    signal_dict: Dict[str, ndarray] = {
        name: np.full(
            (len(xs), *tuple(len(t) for t in block[1])), np.nan, dtype=complex
        )
        for name, block in measure_blocks.items()
    }

    with liveplotter as viewer:
        try:
            xs_tqdm = tqdm(xs, smoothing=0, disable=not progress)
            avgs_tqdm = tqdm(total=cfg["rounds"], smoothing=0, disable=not progress)
            with AsyncFunc(viewer.update) as async_draw:
                assert async_draw is not None
                for i, x in enumerate(xs_tqdm):
                    updateCfg(cfg, i, x)

                    GlobalDeviceManager.setup_devices(cfg["dev"], progress=False)

                    for name, block in measure_blocks.items():
                        (
                            measure_fn,
                            ticks,
                            raw2signal,
                            signal2real,
                            realsignal_callback,
                        ) = block

                        if raw2signal is None:
                            raw2signal = avg_as_signal
                        if signal2real is None:
                            signal2real = take_signal_abs

                        avgs_tqdm.total = cfg["rounds"]
                        avgs_tqdm.reset()
                        avgs_tqdm.refresh()

                        temp_signal = signal_dict[name].copy()

                        def callback(ir: int, *args, **kwargs) -> None:
                            nonlocal temp_signal, avgs_tqdm
                            avgs_tqdm.update(max(ir - avgs_tqdm.n, 0))
                            avgs_tqdm.refresh()

                            temp_signal[i] = raw2signal(ir, *args, **kwargs)
                            real_signal = signal2real(temp_signal)
                            viewer.update(name, xs, *ticks, real_signal)

                            if realsignal_callback is not None:
                                realsignal_callback(i, ir, real_signal)

                        signal_dict[name][i] = measure_fn(cfg, i, signal_dict, callback)

                        avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
                        avgs_tqdm.refresh()

                        async_draw(name, xs, *ticks, signal2real(signal_dict[name]))

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
            for name, block in measure_blocks.items():
                _, ticks, _, signal2real, _ = block
                viewer.update(name, xs, *ticks, signal2real(signal_dict[name]))
        except Exception:
            print("Error during measurement:")
            print_traceback()
        finally:
            xs_tqdm.close()
            avgs_tqdm.close()

    return signal_dict

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm
from zcu_tools.analysis import minus_background
from zcu_tools.program.v2 import MyProgramV2
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow1D, InstantShow2D
from zcu_tools.tools import AsyncFunc, print_traceback


def default_result2signals(
    avg_d: list, std_d: list
) -> Tuple[ndarray, Optional[ndarray]]:
    """
    Convert raw measurement results to complex signals and standard deviations.

    Args:
        avg_d: List containing average measurement data
        std_d: List containing standard deviation data

    Returns:
        Tuple containing:
            - Complex signal array (combining I and Q quadratures)
            - Standard deviation array (maximum across dimensions)
    """
    avg_d = avg_d[0][0].dot([1, 1j])
    std_d = np.max(std_d[0][0], axis=-1)

    return avg_d, std_d


def default_signal2real(signals: ndarray) -> ndarray:
    """
    Convert complex signals to real values by taking the absolute value.

    Args:
        signals: Array of complex signal values

    Returns:
        Array of real values (magnitudes of complex signals)
    """
    return np.abs(signals)


def std2err(stds: Optional[ndarray], N: int) -> Optional[ndarray]:
    """
    Convert standard deviation to standard error by dividing by square root of N.

    Args:
        stds: Array of standard deviations (can be None)
        N: Number of measurements

    Returns:
        Array of standard errors or None if stds is None
    """
    if stds is None:
        return None
    return stds / np.sqrt(N)


def raw2result(ir, sum_d, sum2_d) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw measurement sums to average and standard deviation.

    Args:
        ir: Current repetition index
        sum_d: List of cumulative sums of measurements
        sum2_d: List of cumulative sums of squared measurements

    Returns:
        Tuple containing:
            - List of average values
            - List of standard deviation values
    """
    avg_d = [d / (ir + 1) for d in sum_d]
    std_d = [np.sqrt(d2 / (ir + 1) - d**2) for d, d2 in zip(avg_d, sum2_d)]
    return avg_d, std_d


def calculate_snr1d(signals):
    signals = minus_background(signals)
    m_signals = gaussian_filter1d(signals, sigma=1)

    amps = np.abs(m_signals)
    noise_amps = np.abs(signals - m_signals)

    # use avg of highest three point as signal contrast
    max1_idx = np.argmax(amps)
    max1, amps[max1_idx] = amps[max1_idx], 0
    max2_idx = np.argmax(amps)
    max2, amps[max2_idx] = amps[max2_idx], 0
    max3_idx = np.argmax(amps)
    max3 = amps[max3_idx]

    contrast = (max1 + max2 + max3) / 3

    return contrast / noise_amps.mean()


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
    earlystop_snr: Optional[float] = None,
    **kwargs,
) -> Tuple[MyProgramV2, ndarray]:
    """
    Template for hardware-based parameter sweeps in measurements.

    Args:
        soc: Socket object for hardware communication
        soccfg: Socket configuration
        cfg: Measurement configuration dictionary
        prog_cls: Program class for measurement execution
        ticks: Tuple of arrays representing parameter sweep points
        xlabel: Label for x-axis in visualization
        ylabel: Label for y-axis in visualization
        result2signals: Function to convert raw results to signal data
        signal2real: Function to convert complex signals to real values
        progress: Whether to show progress bars
        **kwargs: Additional arguments passed to the acquire method

    Returns:
        Tuple containing:
            - Program instance
            - Complex signal array from measurements
    """
    signals = np.full(tuple(len(t) for t in ticks), np.nan, dtype=complex)
    stds = np.full_like(signals, np.nan, dtype=float)

    if earlystop_snr is not None and len(ticks) != 1:
        raise ValueError("Early stopping SNR only supports 1D sweep")

    reps = cfg["reps"]

    ViewerCls = [InstantShow1D, InstantShow2D][len(ticks) - 1]

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog = None
    title = None
    with ViewerCls(*ticks, xlabel, ylabel, title=title) as viewer:

        def callback(ir, sum_d, sum2_d):
            nonlocal signals, stds
            signals, stds = result2signals(*raw2result(ir, sum_d, sum2_d))
            if earlystop_snr is not None:
                snr = calculate_snr1d(signals)
                if snr > earlystop_snr:
                    prog.set_early_stop()
                title = f"Current SNR: {snr:.2f}"
            viewer.update_show(
                signal2real(signals), errs=std2err(stds, (ir + 1) * reps), title=title
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
                signal2real(signals),
                errs=std2err(stds, cfg["soft_avgs"] * reps),
                title=title,
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
    """
    Template for 1D software-based parameter sweeps in measurements.

    Args:
        soc: Socket object for hardware communication
        soccfg: Socket configuration
        cfg: Measurement configuration dictionary
        prog_cls: Program class for measurement execution
        xs: Array of x-axis parameter values to sweep
        updateCfg: Function to update configuration for each step
        xlabel: Label for x-axis in visualization
        ylabel: Label for y-axis in visualization
        result2signals: Function to convert raw results to signal data
        signal2real: Function to convert complex signals to real values
        progress: Whether to show progress bars
        **kwargs: Additional arguments passed to the acquire method

    Returns:
        Tuple containing:
            - Array of x-axis parameters
            - Complex signal array from measurements
    """
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
    earlystop_snr: Optional[float] = None,
    **kwargs,
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Template for 2D parameter sweeps with software and hardware components.

    Args:
        soc: Socket object for hardware communication
        soccfg: Socket configuration
        cfg: Measurement configuration dictionary
        prog_cls: Program class for measurement execution
        xs: Array of x-axis parameter values to sweep (software)
        ys: Array of y-axis parameter values (hardware)
        updateCfg: Function to update configuration for each x step
        xlabel: Label for x-axis in visualization
        ylabel: Label for y-axis in visualization
        result2signals: Function to convert raw results to signal data
        signal2real: Function to convert complex signals to real values
        progress: Whether to show progress bars
        earlystop_snr: Optional signal-to-noise ratio for early stopping
        **kwargs: Additional arguments passed to the acquire method

    Returns:
        Tuple containing:
            - Program instance
            - 2D complex signal array from measurements
    """
    cfg = deepcopy(cfg)  # prevent in-place modification
    signals2D = np.full((len(xs), len(ys)), np.nan, dtype=complex)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])  # set initial flux

    title = None
    with InstantShow2D(xs, ys, xlabel, ylabel, title=title) as viewer:
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
                        nonlocal _signals2D, title, avgs_tqdm
                        avgs_tqdm.update(max(ir + 1 - avgs_tqdm.n, 0))
                        avgs_tqdm.refresh()

                        _signals2D[i], _ = result2signals(
                            *raw2result(ir, sum_d, sum2_d)
                        )
                        if earlystop_snr is not None:
                            snr = calculate_snr1d(_signals2D[i])
                            if snr > earlystop_snr:
                                prog.set_early_stop()
                            title = f"Current SNR: {snr:.2f}"
                        viewer.update_show(signal2real(_signals2D), title=title)

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
            viewer.update_show(signal2real(signals2D), ticks=(xs, ys), title=title)
            xs_tqdm.close()
            avgs_tqdm.close()

    return prog, signals2D

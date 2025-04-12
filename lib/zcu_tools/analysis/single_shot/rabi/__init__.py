import threading
import time

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from ...fitting import batch_fit_dual_gauss, fit_gauss, gauss_func
from ...tools import rotate2real

NUM_BINS = 201


def singleshot_rabi_analysis(xs, signals, pure_peak=False):
    signals, angle = rotate2real(signals, ret_angle=True)
    signals = signals.real
    mean_signals = np.mean(signals, axis=1)

    bins = np.linspace(signals.min(), signals.max(), NUM_BINS)

    print("Calculating histogram...", end="")

    list_xdata = [bins[:-1]] * len(xs)
    list_ydata = [np.histogram(signals[i], bins=bins)[0] for i in range(len(xs))]

    print("Fitting dual Gaussian...", end="")

    if pure_peak:
        max_peak = np.argmax(mean_signals)
        min_peak = np.argmin(mean_signals)

        peak1_params, _ = fit_gauss(list_xdata[max_peak], list_ydata[max_peak])
        peak2_params, _ = fit_gauss(list_xdata[min_peak], list_ydata[min_peak])

        if peak1_params[1] > peak2_params[1]:  # make first peak left
            peak1_params, peak2_params = peak2_params, peak1_params

        list_init_p0 = [(*peak1_params, *peak2_params)] * len(xs)
        fixedparams = [None, peak1_params[1], None, None, peak2_params[1], None]
    else:
        list_init_p0 = None
        fixedparams = None

    list_params, _ = batch_fit_dual_gauss(
        list_xdata, list_ydata, list_init_p0=list_init_p0, fixedparams=fixedparams
    )
    list_params = np.array(list_params)

    n_g = list_params[:, 0] * list_params[:, 2] * np.sqrt(2 * np.pi)
    n_e = list_params[:, 3] * list_params[:, 5] * np.sqrt(2 * np.pi)

    n_g /= signals.shape[1] * (bins[1] - bins[0])  # convert to population
    n_e /= signals.shape[1] * (bins[1] - bins[0])  # convert to population

    print("Plotting...")

    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Rotation angle: {180 * angle / np.pi:.3f} deg")

    max_g_idx = np.argmax(n_g)
    fit_gg = gauss_func(bins, *list_params[max_g_idx, :3])
    fit_ge = gauss_func(bins, *list_params[max_g_idx, 3:])
    max_e_idx = np.argmax(n_e)
    fit_eg = gauss_func(bins, *list_params[max_e_idx, :3])
    fit_ee = gauss_func(bins, *list_params[max_e_idx, 3:])

    ax11.hist(signals[max_g_idx], bins=bins)
    ax11.plot(bins, fit_gg + fit_ge, "k-", label="total")
    ax11.plot(bins, fit_eg + fit_ee, "k--")
    ax11.plot(bins, fit_gg, "b-", label="left")
    ax11.plot(bins, fit_ge, "r-", label="right")
    ax11.set_title(f"max left population (x={xs[max_g_idx]:.3f})")
    ax11.legend()

    ax12.hist(signals[max_e_idx], bins=bins)
    ax12.plot(bins, fit_eg + fit_ee, "k-", label="total")
    ax12.plot(bins, fit_gg + fit_ge, "k--")
    ax12.plot(bins, fit_eg, "b-", label="left")
    ax12.plot(bins, fit_ee, "r-", label="right")
    ax12.set_title(f"max right population (x={xs[max_e_idx]:.3f})")
    ax12.legend()

    ax21.plot(xs, n_g, label="left")
    ax21.plot(xs, n_e, label="right")
    ax21.plot(xs, n_g + n_e, label="total")

    ax21.set_ylabel("population")
    ax21.set_ylim(0, 1.01)

    ax22.plot(xs, mean_signals, label="mean")

    ax21.legend()
    ax22.legend()
    plt.show()

    return n_g, n_e, list_params


def visualize_singleshot_rabi(xs, signals, list_params):
    """
    Visualize the results of a singleshot Rabi analysis with optimized performance.
    Shows an interactive widget where you can adjust the slider to display
    signal distributions and corresponding fitting results at different x values.

    Before first user interaction, the slider automatically moves back and forth
    with a 10-second period.
    """

    # Pre-process data - do these calculations only once
    signals_real = rotate2real(signals).real
    mean_signals = np.mean(signals_real, axis=1)
    bins = np.linspace(signals_real.min(), signals_real.max(), NUM_BINS)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Pre-calculate histograms for all x values to avoid recalculation
    histograms = [np.histogram(signals_real[i], bins=bins)[0] for i in range(len(xs))]

    # Pre-calculate all Gaussian fits
    gaussian_fits = []
    for i in range(len(xs)):
        n_g = gauss_func(bin_centers, *list_params[i, :3])
        n_e = gauss_func(bin_centers, *list_params[i, 3:])
        gaussian_fits.append((n_g, n_e))

    # Create fixed figure and axes to reuse
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Containers for plot elements
    hist_bars = None
    marker_line = None

    # Flag to track if user has interacted manually (not from auto-move)
    user_interacted = False
    auto_move_active = True
    programmatic_change = False  # 標記由程序自動改變的值

    def visualize_rabi(x):
        nonlocal hist_bars, marker_line

        # Find closest x value
        idx = np.argmin(np.abs(x - xs))
        actual_x = xs[idx]

        # Clear previous plots for reuse
        ax1.clear()
        if marker_line:
            marker_line.remove()
            marker_line = None

        # Plot histogram using pre-calculated data
        hist_bars = ax1.bar(bin_centers, histograms[idx], width=(bins[1] - bins[0]))

        # Get pre-calculated Gaussian fits
        n_g, n_e = gaussian_fits[idx]

        # Plot fitted distributions
        ax1.plot(bin_centers, n_g + n_e, "k-", linewidth=2, label="Total")
        ax1.plot(bin_centers, n_g, "b-", linewidth=1.5, label="Ground")
        ax1.plot(bin_centers, n_e, "r-", linewidth=1.5, label="Excited")

        # Set up axes
        ax1.set_title(f"x = {actual_x:.3f}")
        ax1.legend(frameon=False)

        # Only redraw the vertical line in the second plot
        marker_line = ax2.axvline(x=actual_x, color="r", linestyle="--")

        # Use draw_idle() for more efficient redraws
        fig.canvas.draw_idle()

    # Set up the second plot once (doesn't change except for marker line)
    ax2.plot(xs, mean_signals, "o-", markersize=3, color="blue", alpha=0.7)
    ax2.set_title("Rabi Oscillation")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Signal")

    # Optimize matplotlib settings for speed
    plt.rcParams["path.simplify"] = True
    plt.rcParams["path.simplify_threshold"] = 1.0
    mpl.style.use("fast")  # Use the fast style

    # Create slider
    scroll_bar = widgets.FloatSlider(
        value=xs[0],
        min=xs.min(),
        max=xs.max(),
        step=(xs.max() - xs.min()) / min(50, len(xs)),
        description="x",
        continuous_update=False,
    )

    # Output widget to display plots
    out = widgets.Output()

    # When user manually changes slider value
    def on_value_change(change):
        nonlocal user_interacted, programmatic_change

        # 只有當不是程序自動改變時，才認為是用戶交互
        if not programmatic_change:
            user_interacted = True

        with out:
            out.clear_output(wait=True)
            visualize_rabi(change["new"])

    # Register slider value change handler
    scroll_bar.observe(on_value_change, names="value")

    # 監聽圖表的點擊事件
    def on_plot_click(event):
        nonlocal user_interacted
        user_interacted = True

    fig.canvas.mpl_connect("button_press_event", on_plot_click)

    # Initial plot
    with out:
        visualize_rabi(xs[0])

    # Auto-move slider function
    def auto_move_slider():
        nonlocal programmatic_change, auto_move_active

        start_time = time.time()
        period = 10.0  # 10秒周期

        while auto_move_active and not user_interacted:
            elapsed = time.time() - start_time
            # 計算位置 (0到1再回到0的循環)
            position = (elapsed % period) / period

            # 映射到x值範圍
            current_x = xs.min() + position * (xs.max() - xs.min())

            # 標記為程序改變，然後更新滑塊
            programmatic_change = True
            scroll_bar.value = current_x
            programmatic_change = False

            # 小延遲
            time.sleep(0.1)

    # Display widgets
    display(
        widgets.VBox([widgets.Label("拖動滑塊或點擊圖表停止自動移動"), scroll_bar, out])
    )

    # Start auto-move in a separate thread
    auto_thread = threading.Thread(target=auto_move_slider, daemon=True)
    auto_thread.start()

    # 返回一個能夠停止自動移動的函數
    def stop_auto_motion():
        nonlocal user_interacted, auto_move_active
        user_interacted = True
        auto_move_active = False

        time.sleep(1.0)
        plt.close()

    return stop_auto_motion


__all__ = [
    "singleshot_rabi_analysis",
    "visualize_singleshot_rabi",
]

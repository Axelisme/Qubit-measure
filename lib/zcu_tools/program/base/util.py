from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class OnlineStatisticTracker:
    """
    Online statistics tracker for 2D IQ data with support for leading batch dimensions.

    Input shape: (..., m, 2) where:
        - ... : leading dimensions (treated as independent channels)
        - m   : number of samples per update
        - 2   : IQ dimensions (I, Q)

    Example:
        - (100, 2): single channel, 100 samples
        - (2, 100, 2): two independent channels (e.g., G and E), 100 samples each
    """

    def __init__(self) -> None:
        self.n = 0
        self.mean: Optional[NDArray[np.float64]] = None
        self.M2: Optional[NDArray[np.float64]] = None
        self.medians: List[NDArray[np.float64]] = []

    @staticmethod
    def _merge(
        n1: int,
        mean1: NDArray[np.float64],
        M2_1: NDArray[np.float64],
        n2: int,
        mean2: NDArray[np.float64],
        M2_2: NDArray[np.float64],
    ) -> Tuple[int, NDArray[np.float64], NDArray[np.float64]]:
        """Merge two sets of statistics. Supports leading dimensions."""
        n = n1 + n2
        mean = (mean1 * n1 + mean2 * n2) / n

        delta = mean1 - mean2  # (..., 2)
        # Compute outer product for each leading dimension: (..., 2, 2)
        delta_outer = delta[..., :, None] * delta[..., None, :]
        M2 = M2_1 + M2_2 + delta_outer * (n1 * n2 / n)

        return n, mean, M2

    def update(self, points: NDArray[np.float64]) -> None:
        """
        Update statistics with new points.

        Args:
            points: shape (..., m, 2) where m is number of samples
        """
        assert len(points.shape) >= 2
        assert points.shape[-1] == 2, "Last dimension must be 2 (I, Q)"

        cur_n = points.shape[-2]
        assert cur_n >= 1

        points_mean = np.mean(points, axis=-2)  # (..., 2)
        centered_points = points - points_mean[..., None, :]  # (..., m, 2)

        # Compute M2 for each leading dimension: (..., 2, 2)
        # M2 = sum of outer products of centered points
        cur_M2 = np.einsum("...mi,...mj->...ij", centered_points, centered_points)

        cur_median = np.median(points, axis=-2)  # (..., 2)

        # first update
        if self.n == 0:
            self.n = cur_n
            self.mean = points_mean
            self.M2 = cur_M2
            self.medians = [cur_median]
            return

        assert self.mean is not None
        assert self.M2 is not None

        self.n, self.mean, self.M2 = self._merge(
            self.n, self.mean, self.M2, cur_n, points_mean, cur_M2
        )
        self.medians.append(cur_median)

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Returns covariance matrix with shape (..., 2, 2)."""
        if self.M2 is None:
            raise ValueError("Covariance is not available yet")
        if self.n == 1:
            return self.M2  # (..., 2, 2)

        return self.M2 / (self.n - 1)

    @property
    def rough_median(self) -> NDArray[np.float64]:
        """Returns rough median with shape (..., 2)."""
        return np.median(self.medians, axis=0)


if __name__ == "__main__":
    from copy import copy

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    def plot_covariance_ellipse(
        ax, mean: NDArray, cov: NDArray, n_std: float = 2.0, **kwargs
    ):
        """Plot an ellipse representing the covariance matrix."""
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort by eigenvalue (largest first)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Calculate angle of ellipse
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # Width and height are 2 * n_std * sqrt(eigenvalue)
        width, height = 2 * n_std * np.sqrt(eigenvalues)

        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellipse)
        return ellipse

    # Generate test data: Two bimodal distributions G and E (simulating qubit readout)
    # Using leading dimension to track G and E independently
    np.random.seed(42)

    # Two isotropic Gaussians with different centers, same std
    mean1 = np.array([2.0, 3.0])  # Center of first Gaussian
    mean2 = np.array([6.0, 7.0])  # Center of second Gaussian
    std = 0.8  # Same standard deviation for both
    iso_cov = np.eye(2) * std**2

    # G distribution: weight (0.7, 0.3) - mostly at peak1
    # E distribution: weight (0.2, 0.8) - mostly at peak2
    weight_G = (0.7, 0.3)
    weight_E = (0.2, 0.8)

    # Generate samples in batches to test online update
    n_batches = 50
    batch_size = 100
    all_points_G = []
    all_points_E = []

    # Single tracker for both G and E (using leading dimension)
    tracker = OnlineStatisticTracker()

    for i in range(n_batches):
        # Generate G batch (weight 0.7, 0.3)
        n1_G = int(batch_size * weight_G[0])
        n2_G = batch_size - n1_G
        samples1_G = np.random.multivariate_normal(mean1, iso_cov, size=n1_G)
        samples2_G = np.random.multivariate_normal(mean2, iso_cov, size=n2_G)
        batch_G = np.vstack([samples1_G, samples2_G])
        np.random.shuffle(batch_G)

        # Generate E batch (weight 0.2, 0.8)
        n1_E = int(batch_size * weight_E[0])
        n2_E = batch_size - n1_E
        samples1_E = np.random.multivariate_normal(mean1, iso_cov, size=n1_E)
        samples2_E = np.random.multivariate_normal(mean2, iso_cov, size=n2_E)
        batch_E = np.vstack([samples1_E, samples2_E])
        np.random.shuffle(batch_E)

        all_points_G.append(batch_G)
        all_points_E.append(batch_E)

        # Stack G and E as leading dimension: shape (2, batch_size, 2)
        # Index 0 = G, Index 1 = E
        batch_GE = np.stack([batch_G, batch_E], axis=0)
        tracker.update(batch_GE)

    # Combine all points for comparison
    all_points_G = np.vstack(all_points_G)
    all_points_E = np.vstack(all_points_E)

    # Calculate statistics using numpy directly
    numpy_mean_G = np.mean(all_points_G, axis=0)
    numpy_median_G = np.median(all_points_G, axis=0)
    numpy_cov_G = np.cov(all_points_G.T)

    numpy_mean_E = np.mean(all_points_E, axis=0)
    numpy_median_E = np.median(all_points_E, axis=0)
    numpy_cov_E = np.cov(all_points_E.T)

    # Get online statistics (shape: (2, 2) for mean/median, (2, 2, 2) for cov)
    online_mean = tracker.mean  # (2, 2) - [G, E] x [I, Q]
    online_median = tracker.rough_median  # (2, 2)
    online_cov = tracker.covariance  # (2, 2, 2) - [G, E] x [2, 2]

    # Extract G and E statistics
    online_mean_G, online_mean_E = online_mean[0], online_mean[1]
    online_median_G, online_median_E = online_median[0], online_median[1]
    online_cov_G, online_cov_E = online_cov[0], online_cov[1]

    # Print comparison
    print("=" * 70)
    print("Two Bimodal Distributions (G and E) - Independent Tracking")
    print(f"  Peak 1: {mean1}, Peak 2: {mean2}, Shared std: {std}")
    print(f"  G distribution: weights = {weight_G} (mostly peak1)")
    print(f"  E distribution: weights = {weight_E} (mostly peak2)")
    print(f"  n_batches={n_batches}, batch_size={batch_size}")
    print(f"  Input shape per update: (2, {batch_size}, 2) = [G/E, samples, I/Q]")
    print()
    print("=== G Distribution (Red) ===")
    print(f"  NumPy Mean:    {numpy_mean_G}")
    print(f"  Online Mean:   {online_mean_G}")
    print(f"  Mean diff:     {online_mean_G - numpy_mean_G}")
    print(f"  NumPy Median:  {numpy_median_G}")
    print(f"  Online Median: {online_median_G}")
    print(f"  Median diff:   {online_median_G - numpy_median_G}")
    print()
    print("=== E Distribution (Blue) ===")
    print(f"  NumPy Mean:    {numpy_mean_E}")
    print(f"  Online Mean:   {online_mean_E}")
    print(f"  Mean diff:     {online_mean_E - numpy_mean_E}")
    print(f"  NumPy Median:  {numpy_median_E}")
    print(f"  Online Median: {online_median_E}")
    print(f"  Median diff:   {online_median_E - numpy_median_E}")
    print("=" * 70)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: scatter plot with G (red) and E (blue) distributions
    ax1 = axes[0]
    ax1.scatter(
        all_points_G[:, 0],
        all_points_G[:, 1],
        alpha=0.3,
        s=10,
        color="red",
        label=f"G ({weight_G[0]:.0%},{weight_G[1]:.0%})",
    )
    ax1.scatter(
        all_points_E[:, 0],
        all_points_E[:, 1],
        alpha=0.3,
        s=10,
        color="blue",
        label=f"E ({weight_E[0]:.0%},{weight_E[1]:.0%})",
    )
    # Plot the two Gaussian centers
    ax1.scatter(
        *mean1,
        color="darkgreen",
        s=200,
        marker="x",
        linewidths=3,
        label="Peak 1",
        zorder=5,
    )
    ax1.scatter(
        *mean2,
        color="lime",
        s=200,
        marker="x",
        linewidths=3,
        label="Peak 2",
        zorder=5,
    )
    # Plot statistics for G
    ax1.scatter(
        *numpy_mean_G,
        color="darkred",
        s=100,
        marker="+",
        linewidths=2,
        label="G mean",
        zorder=5,
    )
    ax1.scatter(
        *numpy_median_G,
        color="darkred",
        s=100,
        marker="s",
        facecolors="none",
        linewidths=2,
        label="G median",
        zorder=5,
    )
    # Plot statistics for E
    ax1.scatter(
        *numpy_mean_E,
        color="darkblue",
        s=100,
        marker="+",
        linewidths=2,
        label="E mean",
        zorder=5,
    )
    ax1.scatter(
        *numpy_median_E,
        color="darkblue",
        s=100,
        marker="s",
        facecolors="none",
        linewidths=2,
        label="E median (NumPy)",
        zorder=5,
    )
    # Plot online statistics for G (circles)
    ax1.scatter(
        *online_mean_G,
        color="red",
        s=150,
        marker="o",
        facecolors="none",
        linewidths=2,
        label="G mean (Online)",
        zorder=6,
    )
    ax1.scatter(
        *online_median_G,
        color="orange",
        s=150,
        marker="D",
        facecolors="none",
        linewidths=2,
        label="G median (Online)",
        zorder=6,
    )
    # Plot online statistics for E (circles)
    ax1.scatter(
        *online_mean_E,
        color="blue",
        s=150,
        marker="o",
        facecolors="none",
        linewidths=2,
        label="E mean (Online)",
        zorder=6,
    )
    ax1.scatter(
        *online_median_E,
        color="cyan",
        s=150,
        marker="D",
        facecolors="none",
        linewidths=2,
        label="E median (Online)",
        zorder=6,
    )

    # Plot covariance ellipses for each Gaussian center (2 std)
    plot_covariance_ellipse(
        ax1,
        mean1,
        iso_cov,
        n_std=2,
        fill=False,
        color="darkgreen",
        linestyle="-",
        linewidth=2,
        label="Peak 1 (2σ)",
    )
    plot_covariance_ellipse(
        ax1,
        mean2,
        iso_cov,
        n_std=2,
        fill=False,
        color="lime",
        linestyle="-",
        linewidth=2,
        label="Peak 2 (2σ)",
    )
    # Plot online covariance ellipses for G and E
    plot_covariance_ellipse(
        ax1,
        online_mean_G,
        online_cov_G,
        n_std=2,
        fill=False,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="G cov (Online 2σ)",
    )
    plot_covariance_ellipse(
        ax1,
        online_mean_E,
        online_cov_E,
        n_std=2,
        fill=False,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label="E cov (Online 2σ)",
    )

    ax1.set_xlabel("X (I)")
    ax1.set_ylabel("Y (Q)")
    ax1.set_title("G (Red) and E (Blue) Distributions - Qubit Readout Simulation")
    ax1.legend(loc="upper left", fontsize=7, ncol=2)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Right plot: convergence of online statistics for G and E independently
    ax2 = axes[1]

    # Track statistics over cumulative updates using leading dimension
    tracker_conv = OnlineStatisticTracker()
    means_history_G = []
    means_history_E = []
    medians_history_G = []
    medians_history_E = []
    n_history = []

    # Re-generate batches with same seed for convergence plot
    np.random.seed(42)
    for i in range(n_batches):
        # Generate G batch
        n1_G = int(batch_size * weight_G[0])
        n2_G = batch_size - n1_G
        samples1_G = np.random.multivariate_normal(mean1, iso_cov, size=n1_G)
        samples2_G = np.random.multivariate_normal(mean2, iso_cov, size=n2_G)
        batch_G = np.vstack([samples1_G, samples2_G])
        np.random.shuffle(batch_G)

        # Generate E batch
        n1_E = int(batch_size * weight_E[0])
        n2_E = batch_size - n1_E
        samples1_E = np.random.multivariate_normal(mean1, iso_cov, size=n1_E)
        samples2_E = np.random.multivariate_normal(mean2, iso_cov, size=n2_E)
        batch_E = np.vstack([samples1_E, samples2_E])
        np.random.shuffle(batch_E)

        # Stack G and E as leading dimension: shape (2, batch_size, 2)
        batch_GE = np.stack([batch_G, batch_E], axis=0)
        tracker_conv.update(batch_GE)

        # Record history for G (index 0) and E (index 1)
        means_history_G.append(tracker_conv.mean[0].copy())
        means_history_E.append(tracker_conv.mean[1].copy())
        medians_history_G.append(tracker_conv.rough_median[0].copy())
        medians_history_E.append(tracker_conv.rough_median[1].copy())
        n_history.append(tracker_conv.n)

    means_history_G = np.array(means_history_G)
    means_history_E = np.array(means_history_E)
    medians_history_G = np.array(medians_history_G)
    medians_history_E = np.array(medians_history_E)
    n_history = np.array(n_history)

    # Plot G convergence (red tones)
    ax2.plot(n_history, means_history_G[:, 0], "r-", label="G Mean I", linewidth=2)
    ax2.plot(
        n_history, means_history_G[:, 1], "r-", label="G Mean Q", linewidth=2, alpha=0.6
    )
    ax2.plot(
        n_history,
        medians_history_G[:, 0],
        "r--",
        label="G Median I",
        linewidth=1.5,
        alpha=0.7,
    )
    ax2.plot(
        n_history,
        medians_history_G[:, 1],
        "r--",
        label="G Median Q",
        linewidth=1.5,
        alpha=0.4,
    )

    # Plot E convergence (blue tones)
    ax2.plot(n_history, means_history_E[:, 0], "b-", label="E Mean I", linewidth=2)
    ax2.plot(
        n_history, means_history_E[:, 1], "b-", label="E Mean Q", linewidth=2, alpha=0.6
    )
    ax2.plot(
        n_history,
        medians_history_E[:, 0],
        "b--",
        label="E Median I",
        linewidth=1.5,
        alpha=0.7,
    )
    ax2.plot(
        n_history,
        medians_history_E[:, 1],
        "b--",
        label="E Median Q",
        linewidth=1.5,
        alpha=0.4,
    )

    # Show peak centers as reference lines
    ax2.axhline(
        mean1[0],
        color="darkgreen",
        linestyle=":",
        alpha=0.6,
        label=f"Peak 1 I={mean1[0]}",
    )
    ax2.axhline(mean1[1], color="darkgreen", linestyle=":", alpha=0.4)
    ax2.axhline(
        mean2[0], color="lime", linestyle=":", alpha=0.6, label=f"Peak 2 I={mean2[0]}"
    )
    ax2.axhline(mean2[1], color="lime", linestyle=":", alpha=0.4)

    ax2.set_xlabel("Number of samples per channel")
    ax2.set_ylabel("Value")
    ax2.set_title("Convergence of G (Red) and E (Blue) - Independent Tracking")
    ax2.legend(loc="center right", fontsize=6, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

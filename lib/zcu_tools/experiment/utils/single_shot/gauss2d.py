from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sklearn.mixture import GaussianMixture


def gauss_2d(
    x: np.ndarray, y: np.ndarray, x0: float, y0: float, sigma: float, n: float
) -> np.ndarray:
    return n * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def fit_gauss_2d(
    xs: np.ndarray, ys: np.ndarray, num_gauss: int = 2
) -> Tuple[np.ndarray, "GaussianMixture"]:
    """
    擬合給定數量的二維各向同性高斯分佈到採樣點。
    使用 GaussianMixture 內建的 k-means 初始化。

    參數:
        xs (ndarray): 形狀為 (N,) 的x坐標。
        ys (ndarray): 形狀為 (N,) 的y坐標。
        num_gauss (int): 要擬合的高斯分佈數量，預設為 2。

    返回:
        params (ndarray): 形狀為 (num_gauss, 4) 的擬合結果。
                          每一行代表一個高斯分佈的參數 [x0, y0, sigma, n]:
                          x0, y0: 中心坐標
                          sigma: 標準差 (各向同性)
                          n: population (和為1)
        gmm (GaussianMixture): 擬合的模型對象。
                          可以用於進一步分析或預測。
    """

    from sklearn.mixture import GaussianMixture

    # 使用 GaussianMixture (EM算法) 優化參數
    # covariance_type='spherical' 表示每個組件有自己的單一方差 (sigma^2)，即各向同性。
    # init_params='kmeans' (預設) 會使用 k-means 初始化 GMM 參數。
    # n_init 控制 k-means 初始化和 EM 運行的次數，選擇最佳結果。
    #   增加 n_init 可以提高找到更好局部最優解的機會。
    gmm = GaussianMixture(
        n_components=num_gauss,
        covariance_type="spherical",
        init_params="kmeans",  # 顯式指定，儘管是預設值
        n_init=10,  # 運行多次初始化以獲得更穩定的結果 (預設是1)
        random_state=42,  # 確保 GMM 初始化和擬合過程中的隨機性可重複
        reg_covar=1e-6,  # 增加穩定性，防止協方差矩陣奇異
        tol=1e-4,  # 收斂閾值
        max_iter=200,  # 最大迭代次數
    )

    try:
        gmm.fit(np.vstack((xs, ys)).T)
    except ValueError as e:
        # 例如，如果所有點都相同，並且 reg_covar 不足以使其非奇異
        print(
            f"Error during GMM fitting: {e}. This can happen with degenerate data "
            "or insufficient points for the number of components. "
            "Returning zero parameters."
        )
        return np.zeros((num_gauss, 4)), gmm

    if not gmm.converged_:
        print("Warning: GaussianMixture did not converge.")

    # 填充實際擬合的組件參數
    means = gmm.means_
    weights = gmm.weights_
    covariances = cast(NDArray[np.float64], gmm.covariances_)
    covariances = np.clip(covariances, 1e-9, None)  # 防止 sqrt(負數) 或 sigma=0

    assert weights is not None
    assert means is not None

    sigmas = np.sqrt(covariances)

    params = np.column_stack((means[:, 0], means[:, 1], sigmas, weights))

    return params, gmm


def fit_gauss_2d_bayesian(xs: np.ndarray, ys: np.ndarray, num_gauss: int = 3):
    """
    使用 BayesianGaussianMixture 擬合二維各向同性高斯分佈。
    模型會自動決定實際使用的成分數。

    參數:
        xs (ndarray): 形狀為 (N,) 的x坐標。
        ys (ndarray): 形狀為 (N,) 的y坐標。
        num_gauss (int): 最多允許的高斯分佈數量，預設為 3。

    返回:
        params (ndarray): 形狀為 (num_gauss, 4) 的擬合結果。
                          每一行代表一個高斯分佈的參數 [x0, y0, sigma, n]。
        bgmm (BayesianGaussianMixture): 擬合的模型對象。
                          可以用於進一步分析或預測。
    """
    from sklearn.mixture import BayesianGaussianMixture

    bgmm = BayesianGaussianMixture(
        n_components=num_gauss,
        covariance_type="spherical",
        init_params="kmeans",
        n_init=num_gauss,
        random_state=42,
        reg_covar=1e-6,
        weight_concentration_prior_type="dirichlet_process",  # 自動決定 m
        weight_concentration_prior=0.1,  # 越小越稀疏，偏好少量成分
        max_iter=500,
        tol=1e-3,
    )

    try:
        bgmm.fit(np.vstack((xs, ys)).T)
    except ValueError as e:
        print(
            f"Error during Bayesian GMM fitting: {e}. Possibly due to degenerate data. Returning zero parameters."
        )
        return np.zeros((1, 4)), bgmm

    if not bgmm.converged_:
        print("Warning: BayesianGaussianMixture did not converge.")

    means = bgmm.means_
    covariances = np.maximum(bgmm.covariances_, 1e-9)
    sigmas = np.sqrt(covariances)
    weights = bgmm.weights_

    params = np.column_stack((means[:, 0], means[:, 1], sigmas, weights))

    return params, bgmm

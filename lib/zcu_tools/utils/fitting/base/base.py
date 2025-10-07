from copy import deepcopy
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy as sp


def with_fixed_params(
    fitfunc: Callable[..., np.ndarray],
    init_p: List[float],
    bounds: Tuple[List[float], List[float]],
    fixedparams: List[Optional[float]],
) -> Tuple[Callable[..., np.ndarray], np.ndarray, np.ndarray]:
    fixedparams = np.array(fixedparams, dtype=float)
    non_fixed_idxs = np.isnan(fixedparams)

    cur_p = deepcopy(fixedparams)

    def wrapped_func(xs: np.ndarray, *args) -> np.ndarray:
        if len(args) != np.sum(non_fixed_idxs):
            raise ValueError(
                f"Expected {np.sum(non_fixed_idxs)} arguments, got {len(args)}."
            )
        cur_p[non_fixed_idxs] = args  #
        return fitfunc(xs, *cur_p)

    if bounds is not None:
        bounds = np.array(bounds)[:, non_fixed_idxs]

    return wrapped_func, np.array(init_p)[non_fixed_idxs], bounds


def add_fixed_params_back(
    pOpt: List[float], pCov: np.ndarray, fixedparams: List[Optional[float]]
) -> Tuple[List[float], np.ndarray]:
    fixedparams = np.array(fixedparams, dtype=float)
    non_fixed_idxs = np.isnan(fixedparams)

    pOpt_full = fixedparams.copy()
    pOpt_full[non_fixed_idxs] = pOpt

    pCov_full = np.zeros((len(fixedparams), len(fixedparams)))
    pCov_full[:, non_fixed_idxs][non_fixed_idxs] = pCov

    return pOpt_full, pCov_full


def fit_func(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitfunc: Callable[..., np.ndarray],
    init_p: Optional[List[float]] = None,
    bounds: Optional[Tuple[List[float], List[float]]] = None,
    fixedparams: Optional[List[Optional[float]]] = None,
    estimate_sigma: bool = True,
    **kwargs,
) -> Tuple[List[float], np.ndarray]:
    if fixedparams is not None and any([p is not None for p in fixedparams]):
        if init_p is None:
            raise ValueError(
                "Initial parameters must be provided when fixed parameters are specified."
            )

        fitfunc, init_p, bounds = with_fixed_params(
            fitfunc, init_p, bounds, fixedparams
        )

    # estimate the sigma
    if estimate_sigma:
        sigma = np.std(np.diff(ydata)) / np.sqrt(2)
        kwargs.setdefault("sigma", np.full_like(ydata, sigma))
        kwargs.setdefault("absolute_sigma", True)

    try:
        pOpt, pCov = sp.optimize.curve_fit(
            fitfunc, xdata, ydata, p0=init_p, bounds=bounds, **kwargs
        )
    except RuntimeError as e:
        print("Warning: fit failed!")
        print(e)
        pOpt = list(init_p)
        pCov = np.full(shape=(len(init_p), len(init_p)), fill_value=np.inf)

    if fixedparams is not None and len(fixedparams) > 0:
        pOpt, pCov = add_fixed_params_back(pOpt, pCov, fixedparams)

    return pOpt, pCov


def batch_fit_func(
    list_xdata,
    list_ydata,
    fitfunc,
    list_init_p,
    shared_idxs,
    list_bounds=None,
    fixedparams=None,
    **kwargs,
):
    n_groups = len(list_xdata)
    n_params_total = len(list_init_p[0])  # 總參數個數（以第一組為準）

    # 計算哪些是非共享參數索引
    shared_idxs = set(shared_idxs)
    local_idxs = set(range(n_params_total)) - shared_idxs
    local_idxs = sorted(local_idxs)
    shared_idxs = sorted(shared_idxs)

    n_shared = len(shared_idxs)
    n_local = len(local_idxs)

    def build_batch_params(list_p0, list_bounds, fixedparams):
        nonlocal shared_idxs, local_idxs

        shared_p0 = [np.mean([p[j] for p in list_p0]) for j in shared_idxs]
        local_p0 = []
        for p in list_p0:
            local_p0.extend(p[j] for j in local_idxs)
        batch_p0 = shared_p0 + local_p0

        # 組合 bounds（若有）
        if list_bounds is not None:
            list_bounds = np.array(list_bounds)
            lower_shared = np.min(list_bounds[:, 0, shared_idxs], axis=0)
            upper_shared = np.max(list_bounds[:, 1, shared_idxs], axis=0)
            lower_local = []
            upper_local = []
            for b in list_bounds:
                lower_local.extend(b[0][j] for j in local_idxs)
                upper_local.extend(b[1][j] for j in local_idxs)
            batch_bounds = (
                np.concatenate([lower_shared, lower_local]),
                np.concatenate([upper_shared, upper_local]),
            )
        else:
            batch_bounds = None

        if fixedparams is not None and any([p is not None for p in fixedparams]):
            shared_fixed = [fixedparams[s] for s in shared_idxs]
            local_fixed = [fixedparams[ll] for ll in local_idxs] * n_groups
            fixedparams = shared_fixed + local_fixed
        else:
            fixedparams = None

        return batch_p0, batch_bounds, fixedparams

    def build_total_params(batch_params):
        total_indices = []
        total_params = []
        for i in range(n_groups):
            group_indices = [None] * n_params_total
            for j, share_idx in enumerate(shared_idxs):
                group_indices[share_idx] = j
            for j, local_idx in enumerate(local_idxs):
                group_indices[local_idx] = j + i * n_local + n_shared
            total_indices.append(group_indices)
            total_params.append([batch_params[i] for i in group_indices])
        return total_params, total_indices

    # 定義合併的全域模型函數
    def batched_func(_, *batch_params):
        total_params, _ = build_total_params(batch_params)
        return np.concatenate(
            [fitfunc(x, *pi) for x, pi in zip(list_xdata, total_params)]
        )

    # 組合 y 為長向量
    batch_y = np.concatenate(list_ydata)

    batch_p0, batch_bounds, batch_fixedparams = build_batch_params(
        list_init_p, list_bounds, fixedparams
    )

    # 擬合
    popt, pcov = fit_func(
        None,
        batch_y,
        batched_func,
        init_p=batch_p0,
        bounds=batch_bounds,
        fixedparams=batch_fixedparams,
        **kwargs,
    )

    # 還原每組共變異數子矩陣
    list_popt, total_indices = build_total_params(popt)
    list_pcov = []
    for i in range(n_groups):
        # 擷取子矩陣
        sub_pcov = pcov[np.ix_(total_indices[i], total_indices[i])]

        # 建立完整大小的共變異數矩陣（含共享與本地參數）
        pcov_i = np.full((n_params_total, n_params_total), np.nan)
        for m, mi in enumerate(shared_idxs + local_idxs):
            for n, ni in enumerate(shared_idxs + local_idxs):
                pcov_i[mi, ni] = sub_pcov[m, n]

        list_pcov.append(pcov_i)

    return list_popt, list_pcov


def assign_init_p(
    fitparams: List[Optional[float]], init_p: List[float]
) -> List[Optional[float]]:
    for i, p in enumerate(init_p):
        if fitparams[i] is None:
            fitparams[i] = p
    return fitparams


def fit_line(xdata: np.ndarray, ydata: np.ndarray) -> Tuple[float, float]:
    """params: [a, b] -> y = a * x + b"""

    from scipy.stats import linregress

    a, b, *_ = linregress(xdata, ydata)

    return a, b

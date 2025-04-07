from copy import deepcopy

import numpy as np
import scipy as sp


def with_fixed_params(fitfunc, init_p, bounds, fixedparams):
    fixedparams = np.array(fixedparams, dtype=float)
    non_fixed_idxs = ~np.isnan(fixedparams)

    cur_p = deepcopy(fixedparams)

    def wrapped_func(*args):
        if len(args) != np.sum(non_fixed_idxs):
            raise ValueError(
                f"Expected {np.sum(non_fixed_idxs)} arguments, got {len(args)}."
            )
        cur_p[non_fixed_idxs] = args
        return fitfunc(*cur_p)

    if bounds is not None:
        bounds = np.array(bounds)[:, non_fixed_idxs]

    return wrapped_func, init_p[non_fixed_idxs], bounds


def add_fixed_params_back(pOpt, pCov, fixedparams):
    fixedparams = np.array(fixedparams, dtype=float)
    non_fixed_idxs = ~np.isnan(fixedparams)

    pOpt_full = fixedparams.copy()
    pOpt_full[non_fixed_idxs] = pOpt

    pCov_full = np.zeros((len(fixedparams), len(fixedparams)))
    for i, row_idx in enumerate(non_fixed_idxs):
        for j, col_idx in enumerate(non_fixed_idxs):
            pCov_full[row_idx, col_idx] = pCov[i, j]

    return pOpt_full, pCov_full


def fit_func(
    xdata, ydata, fitfunc, init_p=None, bounds=None, fixedparams=None, **kwargs
):
    if fixedparams is not None and len(fixedparams) > 0:
        if init_p is None:
            raise ValueError(
                "Initial parameters must be provided when fixed parameters are specified."
            )

        fitfunc, init_p, bounds = with_fixed_params(
            fitfunc, init_p, bounds, fixedparams
        )

    try:
        pOpt, pCov = sp.optimize.curve_fit(
            fitfunc, xdata, ydata, p0=init_p, bounds=bounds, **kwargs
        )
    except RuntimeError as e:
        print("Warning: fit failed!")
        print(e)
        pOpt = init_p
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
    **kwargs,
):
    n_groups = len(list_xdata)
    n_params_total = len(list_init_p[0])  # 總參數個數（以第一組為準）

    # 計算哪些是非共享參數索引
    all_idxs = set(range(n_params_total))
    shared_idxs = set(shared_idxs)
    local_idxs = sorted(all_idxs - shared_idxs)

    # 排序共享索引確保一致性
    shared_idxs = sorted(shared_idxs)

    # Helper: 根據全域參數展開出單組的參數
    def build_group_params(global_params, group_idx):
        shared_params = global_params[: len(shared_idxs)]
        local_params_flat = global_params[len(shared_idxs) :]
        group_local_params = local_params_flat[
            group_idx * len(local_idxs) : (group_idx + 1) * len(local_idxs)
        ]

        # 把共享與本地參數塞入對應位置
        full_params = [None] * n_params_total
        for i, idx in enumerate(shared_idxs):
            full_params[idx] = shared_params[i]
        for i, idx in enumerate(local_idxs):
            full_params[idx] = group_local_params[i]
        return full_params

    # 定義合併的全域模型函數
    def batched_func(x_concat, *global_params):
        y_concat = []
        for i in range(n_groups):
            xi = list_xdata[i]
            pi = build_group_params(global_params, i)
            yi = fitfunc(xi, *pi)
            y_concat.append(yi)
        return np.concatenate(y_concat)

    # 組合 x/y 為長向量
    batch_x = np.concatenate(list_xdata)
    batch_y = np.concatenate(list_ydata)

    shared_init = [np.mean([p[j] for p in list_init_p]) for j in shared_idxs]
    local_init = []
    for p in list_init_p:
        local_init.extend(p[j] for j in local_idxs)

    batch_p0 = shared_init + local_init

    # 組合 bounds（若有）
    if list_bounds is not None:
        lower_shared = [min(b[0][j] for b in list_bounds) for j in shared_idxs]
        upper_shared = [max(b[1][j] for b in list_bounds) for j in shared_idxs]
        lower_local = []
        upper_local = []
        for b in list_bounds:
            lower_local.extend(b[0][j] for j in local_idxs)
            upper_local.extend(b[1][j] for j in local_idxs)
        batch_bounds = (lower_shared + lower_local, upper_shared + upper_local)
    else:
        batch_bounds = None

    # 擬合
    popt, pcov = fit_func(
        batch_x, batch_y, batched_func, init_p=batch_p0, bounds=batch_bounds, **kwargs
    )

    # 還原每組參數
    list_popt = [build_group_params(popt, i) for i in range(n_groups)]

    # 還原每組共變異數子矩陣
    n_shared = len(shared_idxs)
    n_local = len(local_idxs)
    list_pcov = []

    for i in range(n_groups):
        # 對應這一組參數在 popt 裡的位置
        indices = []
        indices.extend(range(n_shared))  # 共享參數在開頭
        indices.extend(
            range(n_shared + i * n_local, n_shared + (i + 1) * n_local)
        )  # 本組本地參數

        # 擷取子矩陣
        sub_pcov = pcov[np.ix_(indices, indices)]

        # 建立完整大小的共變異數矩陣（含共享與本地參數）
        full_pcov = np.full((n_params_total, n_params_total), np.nan)
        for m, mi in enumerate(shared_idxs + local_idxs):
            for n, ni in enumerate(shared_idxs + local_idxs):
                full_pcov[mi, ni] = sub_pcov[m, n]

        list_pcov.append(full_pcov)

    return list_popt, list_pcov


def assign_init_p(fitparams, init_p):
    for i, p in enumerate(init_p):
        if fitparams[i] is None:
            fitparams[i] = p
    return fitparams


def fit_line(xdata, ydata, fitparams=None):
    def fitfunc(x, a, b):
        return a * x + b

    if fitparams is None:
        fitparams = [None] * 2

    if any([p is None for p in fitparams]):
        a = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        b = ydata[0] - a * xdata[0]
        assign_init_p(fitparams, [a, b])

    bounds = (
        [-np.inf, -np.inf],
        [np.inf, np.inf],
    )

    return fit_func(xdata, ydata, fitfunc, fitparams, bounds)

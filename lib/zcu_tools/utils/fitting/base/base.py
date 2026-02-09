from functools import wraps
from typing import Callable, List, Optional, Sequence, Tuple, cast, TypeVar

import numpy as np
import scipy as sp
from numpy.typing import NDArray

Y_DataType = TypeVar("Y_DataType", bound=np.generic)


def with_fixed_params(
    fitfunc: Callable[..., NDArray[Y_DataType]],
    init_p: Sequence[Optional[float]],
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]],
    fixedparams: Sequence[Optional[float]],
) -> Tuple[
    Callable[..., NDArray[Y_DataType]],
    Sequence[Optional[float]],
    Optional[Tuple[Sequence[float], Sequence[float]]],
]:
    fixedparams_array = np.asarray(fixedparams, dtype=np.float64)  # convert None to nan
    non_fixed_idxs = np.isnan(fixedparams_array)

    @wraps(fitfunc)
    def wrapped_func(xs: NDArray, *args) -> NDArray:
        if len(args) != np.sum(non_fixed_idxs):
            raise ValueError(
                f"Expected {np.sum(non_fixed_idxs)} arguments, got {len(args)}."
            )
        # assign the arguments to the parameters
        params = fixedparams_array.copy()
        params[non_fixed_idxs] = args

        return fitfunc(xs, *params)

    init_p_array = np.array(init_p)[non_fixed_idxs]
    init_p = list(init_p_array)

    if bounds is not None:
        bounds_array = np.array(bounds)[:, non_fixed_idxs]
        bounds = (list(bounds_array[0]), list(bounds_array[1]))
    else:
        bounds = None

    return wrapped_func, init_p, bounds


def add_fixed_params_back(
    pOpt: List[float], pCov: NDArray[np.float64], fixedparams: Sequence[Optional[float]]
) -> Tuple[List[float], NDArray[np.float64]]:
    _fixedparams = np.asarray(fixedparams, dtype=float)
    non_fixed_idxs = np.isnan(_fixedparams)

    pOpt_full = _fixedparams.copy()
    pOpt_full[non_fixed_idxs] = pOpt

    pCov_full = np.zeros((len(_fixedparams), len(_fixedparams)))
    pCov_full[:, non_fixed_idxs][non_fixed_idxs] = pCov

    return list(pOpt_full), pCov_full


def fit_func(
    xdata: NDArray,
    ydata: NDArray[Y_DataType],
    fitfunc: Callable[..., NDArray[Y_DataType]],
    init_p: Optional[Sequence[Optional[float]]] = None,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    fixedparams: Optional[Sequence[Optional[float]]] = None,
    **kwargs,
) -> Tuple[List[float], NDArray[np.float64]]:
    if fixedparams is not None and any([p is not None for p in fixedparams]):
        if init_p is None:
            raise ValueError(
                "Initial parameters must be provided when fixed parameters are specified."
            )

        fitfunc, init_p, bounds = with_fixed_params(
            fitfunc, init_p, bounds, fixedparams
        )

    if bounds is None:
        bounds = (-np.inf, np.inf)  # type: ignore

    try:
        pOpt, pCov = sp.optimize.curve_fit(
            fitfunc, xdata, ydata, p0=init_p, bounds=bounds, **kwargs
        )
    except RuntimeError:
        if init_p is None:
            raise
        pOpt = [p if p is not None else np.nan for p in init_p]
        pCov = np.full(shape=(len(init_p), len(init_p)), fill_value=np.inf)

    if fixedparams is not None and len(fixedparams) > 0:
        pOpt, pCov = add_fixed_params_back(pOpt, pCov, fixedparams)

    return pOpt, pCov


def batch_fit_func(
    list_xdata: List[NDArray],
    list_ydata: List[NDArray[Y_DataType]],
    fitfunc: Callable[..., NDArray[Y_DataType]],
    list_init_p: List[Sequence[float]],
    shared_idxs: List[int],
    list_bounds: Optional[List[Tuple[List[float], List[float]]]] = None,
    fixedparams: Optional[List[Optional[float]]] = None,
    **kwargs,
) -> Tuple[List[List[float]], List[NDArray[np.float64]]]:
    n_groups = len(list_xdata)
    n_params_total = len(list_init_p[0])  # 總參數個數（以第一組為準）

    # 計算哪些是非共享參數索引
    _shared_idxs = set(shared_idxs)
    local_idxs = set(range(n_params_total)) - _shared_idxs
    local_idxs = sorted(local_idxs)
    _shared_idxs = sorted(_shared_idxs)

    n_shared = len(_shared_idxs)
    n_local = len(local_idxs)

    def build_batch_params(
        list_p0: Sequence[Sequence[float]],
        list_bounds: Optional[Sequence[Tuple[Sequence[float], Sequence[float]]]],
        fixedparams: Optional[Sequence[Optional[float]]],
    ) -> Tuple[
        List[float],
        Optional[Tuple[Sequence[float], Sequence[float]]],
        Optional[Sequence[Optional[float]]],
    ]:
        nonlocal _shared_idxs, local_idxs

        shared_p0 = [np.mean([p[j] for p in list_p0]).item() for j in _shared_idxs]
        local_p0 = []
        for p in list_p0:
            local_p0.extend(p[j] for j in local_idxs)
        batch_p0 = shared_p0 + local_p0

        # 組合 bounds（若有）
        if list_bounds is not None:
            array_bounds = np.array(list_bounds)
            lower_shared = np.min(array_bounds[:, 0, _shared_idxs], axis=0)  # type: ignore
            upper_shared = np.max(array_bounds[:, 1, _shared_idxs], axis=0)  # type: ignore
            lower_local = []
            upper_local = []
            for b in list_bounds:
                lower_local.extend(b[0][j] for j in local_idxs)
                upper_local.extend(b[1][j] for j in local_idxs)
            batch_bounds = (
                list(np.concatenate([lower_shared, lower_local])),
                list(np.concatenate([upper_shared, upper_local])),
            )
        else:
            batch_bounds = None

        if fixedparams is not None and any([p is not None for p in fixedparams]):
            shared_fixed = [fixedparams[s] for s in _shared_idxs]
            local_fixed = [fixedparams[ll] for ll in local_idxs] * n_groups
            fixedparams = shared_fixed + local_fixed
        else:
            fixedparams = None

        return batch_p0, batch_bounds, fixedparams

    def build_total_params(
        batch_params: Tuple[float, ...],
    ) -> Tuple[List[List[float]], List[List[int]]]:
        total_indices = []
        total_params = []
        for i in range(n_groups):
            group_indices: List[Optional[int]] = [None] * n_params_total
            for j, share_idx in enumerate(_shared_idxs):
                group_indices[share_idx] = j
            for j, local_idx in enumerate(local_idxs):
                group_indices[local_idx] = j + i * n_local + n_shared
            assert all([gi is not None for gi in group_indices])

            _group_indices = cast(List[int], group_indices)
            total_indices.append(_group_indices)
            total_params.append([batch_params[i] for i in _group_indices])
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
        None,  # type: ignore
        batch_y,
        batched_func,
        init_p=batch_p0,
        bounds=batch_bounds,
        fixedparams=batch_fixedparams,
        **kwargs,
    )

    # 還原每組共變異數子矩陣
    list_popt, total_indices = build_total_params(tuple(popt))
    list_pcov = []
    for i in range(n_groups):
        # 擷取子矩陣
        sub_pcov = pcov[np.ix_(total_indices[i], total_indices[i])]

        # 建立完整大小的共變異數矩陣（含共享與本地參數）
        pcov_i = np.full((n_params_total, n_params_total), np.nan)
        for m, mi in enumerate(_shared_idxs + local_idxs):
            for n, ni in enumerate(_shared_idxs + local_idxs):
                pcov_i[mi, ni] = sub_pcov[m, n]

        list_pcov.append(pcov_i)

    return list_popt, list_pcov


def assign_init_p(
    fitparams: List[Optional[float]], init_p: Sequence[float]
) -> List[Optional[float]]:
    for i, p in enumerate(init_p):
        if fitparams[i] is None:
            fitparams[i] = p
    return fitparams


def fit_line(
    xdata: NDArray[np.float64], ydata: NDArray[np.float64]
) -> Tuple[float, float]:
    """params: [a, b] -> y = a * x + b"""
    a, b, *_ = sp.stats.linregress(xdata, ydata)

    return a, b

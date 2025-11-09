---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt

from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate import flx2mA
```

```python
qub_name = "Q12_2D[3]/Q4"
```

```python
loadpath = f"../result/{qub_name}/params.json"
_, params, mA_c, period, allows, data_dict = load_result(loadpath)
EJ, EC, EL = params

print(allows)

if dispersive_cfg := data_dict.get("dispersive"):
    g = dispersive_cfg["g"]
    r_f = dispersive_cfg["r_f"]
    print(f"g: {g}, r_f: {r_f}")
elif "r_f" in allows:
    r_f = allows["r_f"]
    g = 100e-3
    print(f"r_f: {r_f}")

if "sample_f" in allows:
    sample_f = allows["sample_f"]

r_f = 7.520
g = 100e-3

flxs = np.linspace(-0.05, 0.55, 1000)
```

# Try 2

```python
from typing import Dict, List
import qutip as qt
from tqdm.auto import tqdm
from zcu_tools.simulate.fluxonium.branch.floquet import FloquetBranchAnalysis
from joblib import Parallel, delayed

qub_dim = 20
qub_cutoff = 50
max_photon = 100


def calc_n_crit(
    flx: float, threshold: float, branchs: List[int] = [0, 1]
) -> Dict[int, float]:
    f_analysis = FloquetBranchAnalysis(
        params, r_f, g, flx, qub_dim=qub_dim, qub_cutoff=qub_cutoff
    )

    basis_n0 = f_analysis.make_floquet_basis(photon=0)
    states_n0 = basis_n0.state(t=0)
    last_energies_n0 = {
        b: basis_n0.e_quasi[
            np.argmax(
                [
                    np.abs(state_n0.dag() @ qt.basis(qub_dim, b))
                    for state_n0 in states_n0
                ]
            )
        ]
        for b in branchs
    }

    n = 1
    crit_ns = {b: None for b in branchs}
    while n < max_photon:
        # calculate time average of states
        energies_n = f_analysis.make_floquet_basis(photon=n).e_quasi

        # calculate critical photon number for each branch
        for b in branchs:
            if crit_ns[b] is not None:
                continue  # already found

            last_energy_n0 = last_energies_n0[b]

            # find the closest energy to last_energy_n0
            mod_energies_n = (
                np.mod(energies_n - last_energy_n0 + r_f / 2, r_f) - r_f / 2
            )
            sort_idxs = np.argsort(np.abs(mod_energies_n))
            closest_idx = sort_idxs[1]  # idx 0 is current level
            if np.abs(mod_energies_n[closest_idx]) < threshold:
                crit_ns[b] = n

            last_energies_n0[b] = energies_n[closest_idx]

        # check whether all branches reach critical photon number
        if all(crit_ns[b] is not None for b in branchs):
            break

        n = n + 1
    else:  # reach max photon number
        for b in branchs:
            if crit_ns[b] is None:
                crit_ns[b] = max_photon

    return crit_ns

```

```python
branchs = [0, 1]
threshold = 1e-3

crit_ns_over_flx = Parallel(n_jobs=-1)(
    delayed(calc_n_crit)(flx, threshold, branchs) for flx in tqdm(flxs)
)
crit_ns_over_flx = {
    b: [crit_ns_over_flx[i][b] for i in range(len(flxs))] for b in branchs
}
```

```python
for b in branchs:
    plt.plot(flxs, crit_ns_over_flx[b], label=f"branch {b}")
plt.legend()
plt.show()
```

# Try1

```python
import scqubits as scq  # lazy import

qub_cutoff = 50
qub_dim = 31

fluxonium = scq.Fluxonium(*params, flux=0.5, cutoff=qub_cutoff, truncated_dim=qub_dim)

F_evals = fluxonium.eigenvals(evals_count=qub_dim)
res_dim = int(np.ceil(np.ptp(F_evals) / r_f)) + 1
print("res_dim: ", res_dim)

resonator = scq.Oscillator(r_f, truncated_dim=res_dim)
hilbertspace = scq.HilbertSpace([resonator, fluxonium])
hilbertspace.add_interaction(
    g=g, op1=resonator.creation_operator, op2=fluxonium.n_operator, add_hc=True
)


def update_hilbertspace(sweep_param) -> None:
    fluxonium.flux = sweep_param


scq.settings.PROGRESSBAR_DISABLED = False
sweep = scq.ParameterSweep(
    hilbertspace,
    {"params": flxs},
    update_hilbertspace=update_hilbertspace,
    evals_count=res_dim * qub_dim,
    subsys_update_info={"params": [fluxonium]},
    labeling_scheme="LX",
)

evals = sweep["evals"].toarray()
```

```python
import math


def _solve_positive_quadratic(A: float, B: float, C: float) -> list[float]:
    """
    輔助函數：解 Ax^2 + Bx + C = 0 (假設 A != 0)，
    並返回所有 *正實數* 解的列表。
    """
    roots = []

    # 計算判別式
    discriminant = B**2 - 4 * A * C

    if discriminant < 0:
        # 沒有實數解
        return roots

    sqrt_delta = math.sqrt(discriminant)

    # 計算兩個可能的解
    x1 = (-B + sqrt_delta) / (2 * A)
    x2 = (-B - sqrt_delta) / (2 * A)

    # 檢查 x1 是否為正實數
    if x1 > 0:
        roots.append(x1)

    # 檢查 x2 是否為正實數
    # 如果 discriminant == 0, 則 x1 == x2,
    # 這裡的 'discriminant > 0' 檢查可以避免重複加入
    if discriminant > 0 and x2 > 0:
        roots.append(x2)

    return roots


def solve_min_x(a: float, b: float, c: float, r: float) -> float:
    """
    找出最小的正實數 x0 > 0 使得 p(x0) = ax^2 + bx + c = 0 (mod r)。

    參數:
        a, b, c: 多項式 p(x) = ax^2 + bx + c 的係數 (假設 a != 0)
        r: 模數 (假設 r > 0)

    返回:
        最小的正實數解 x0，如果找不到解，則返回 float('inf')。
    """

    # 候選解列表
    candidates = []

    # --- 1. 檢查 x=0 附近的解 (邊界候選解) ---
    y0 = c  # p(0) = c

    # 找出 p(0) 附近的兩個 k*r 值
    k1 = math.floor(y0 / r)
    k2 = math.ceil(y0 / r)

    # 解方程式: ax^2 + bx + (c - k1*r) = 0
    candidates.extend(_solve_positive_quadratic(a, b, c - k1 * r))

    # 如果 k1 和 k2 不同，才需要解 k2
    if k1 != k2:
        # 解方程式: ax^2 + bx + (c - k2*r) = 0
        candidates.extend(_solve_positive_quadratic(a, b, c - k2 * r))

    # --- 2. 檢查頂點附近的解 (極值候選解) ---

    # 拋物線頂點的 x 座標 (假設 a != 0)
    xv = -b / (2 * a)

    # 只有當頂點在 x > 0 的區域時，才需要檢查
    if xv > 0:
        # 頂點的 y 座標
        yv = a * xv**2 + b * xv + c
        kv = 0.0

        if a > 0:
            # 拋物線開口向上 (yv 是最小值)
            # 我們要找 yv "之後" (向上) 的第一個 k*r
            kv = math.ceil(yv / r)
        else:
            # Read: a < 0, 拋物線開口向下 (yv 是最大值)
            # 我們要找 yv "之後" (向下) 的第一個 k*r
            kv = math.floor(yv / r)

        # 解方程式: ax^2 + bx + (c - kv*r) = 0
        candidates.extend(_solve_positive_quadratic(a, b, c - kv * r))

    # --- 3. 找出最終答案 ---
    if not candidates:
        # 如果列表為空，表示沒有找到任何正實數解
        return float("inf")

    # 返回所有候選解中的最小值
    return min(candidates)
```

```python
def get_Eij(evals, i, j):
    idxs = np.arange(len(flxs))
    return evals[idxs, sweep.dressed_index((i, j)).toarray()]


def calc_n_crit(evals, i, max_n, threshold=1e-2) -> np.ndarray:
    n_crit_j_list = []

    ax_num = int(4 * np.sqrt((qub_dim - 1) / 8))
    fig, axs = plt.subplots(int(np.ceil(ax_num / 2 + 0.1)), ax_num + 1, figsize=(9, 5))
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    ax_count = 0

    ns = np.arange(res_dim - 2)
    for j in range(qub_dim - 1):
        if j == i:
            continue
        T_jis = np.array([get_Eij(evals, n, j) - get_Eij(evals, n, i) for n in ns])
        n_crit_j = []
        T_ji_params = []
        loss_list = []
        for f in range(len(flxs)):
            T_ji_param = np.polyfit(ns, T_jis[:, f], deg=2)
            loss = np.max(np.abs(np.polyval(T_ji_param, ns) - T_jis[:, f]))
            if loss > np.ptp(T_jis[:, f]) * threshold:
                n_crit_j.append(np.nan)
                loss_list.append(0.0)
            else:
                n_crit_j.append(solve_min_x(*T_ji_param, r_f))
                loss_list.append(loss)
            T_ji_params.append(T_ji_param)
            # loss_list.append(loss)
        n_crit_j = np.array(n_crit_j)
        loss_list = np.array(loss_list)

        # remove negative and clip to max_n
        n_crit_j = np.where(n_crit_j < 0, max_n, n_crit_j)
        n_crit_j = np.clip(n_crit_j, 0, max_n)

        n_crit_j_list.append(n_crit_j)

        f = np.argmax(loss_list)
        ax = axs[ax_count // (ax_num + 1), ax_count % (ax_num + 1)]
        ax.plot(
            ns,
            np.polyval(T_ji_params[f], ns),
            label=f"{loss_list[f] / (threshold * np.ptp(T_jis[:, f])):.2f}",
        )
        ax.scatter(ns, T_jis[:, f], s=5)
        ax.legend(fontsize=8)
        ax_count += 1
    plt.show(fig)

    n_crit_j_list = np.array(n_crit_j_list)

    n_crit_js = np.full_like(flxs, np.nan)
    for f in range(len(flxs)):
        if np.all(np.isnan(n_crit_j_list[:, f])):
            continue
        n_crit_js[f] = np.nanmin(n_crit_j_list[:, f])

    return n_crit_js
```

```python
max_n = res_dim * 10
threshold = 1e-3

n_crit_0_flx = calc_n_crit(evals, 0, max_n, threshold)
n_crit_1_flx = calc_n_crit(evals, 1, max_n, threshold)
```

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=flxs, y=n_crit_0_flx, mode="lines", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=flxs, y=n_crit_1_flx, mode="lines", line=dict(color="red")))
fig.show()
```

```python

```

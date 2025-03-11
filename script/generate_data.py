# make energies of fluxonium under different external fluxes
# and save them in a file

import os

import h5py as h5
import numpy as np
import scqubits as scq
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# parameters
data_path = "simulation_data/fluxonium_int.h5"
num_sample = 1000
EJb = (2.0, 6.0)
ECb = (0.8, 2.0)
ELb = (0.01, 0.2)
# EJb = (3.0, 6.5)
# ECb = (0.3, 2.0)
# ELb = (0.5, 3.5)

DRY_RUN = False
scq.settings.PROGRESSBAR_DISABLED = True

cutoff = 50
max_level = 15
flxs = np.linspace(0.0, 0.5, 120)


fluxonium = scq.Fluxonium(1.0, 1.0, 1.0, flux=0.0, cutoff=cutoff)


def calculate_spectrum(flxs, EJ, EC, EL):
    global fluxonium
    fluxonium.EJ = EJ
    fluxonium.EC = EC
    fluxonium.EL = EL
    spectrumData = fluxonium.get_spectrum_vs_paramvals(
        "flux", flxs, evals_count=max_level
    )

    return spectrumData.energy_table


def dump_data(filepath, flxs, params, energies, Ebounds):
    with h5.File(filepath, "w") as f:
        f.create_dataset("Ebounds", data=Ebounds)
        f.create_dataset("flxs", data=flxs)
        f.create_dataset("params", data=params)
        f.create_dataset("energies", data=energies)


def fibonacci_lattice(K):
    """
    在單位球面上生成 K 個近似均勻分布的方向(Fibonacci lattic)）。
    返回: K 個單位向量，形狀為 (K, 3)。
    """
    # 黃金比例
    phi = (1 + np.sqrt(5)) / 2

    # 生成點的索引
    indices = np.arange(K)

    # z 坐標，從 -1 到 1
    z = 1 - (2 * indices + 1) / K

    # 極角 theta 和方位角 phi
    theta = 2 * np.pi * indices / phi
    r = np.sqrt(1 - z**2)  # 球面上的半徑

    # 轉換為笛卡爾坐標
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # 組成單位向量
    directions = np.stack([x, y, z], axis=-1)
    return directions


def fibonacci_lattice_positive_vectorized(K):
    """
    使用 numpy 向量化生成約均勻分布在球面上的點，並篩選出 xyz 坐標都大於 0 的點，
    最後返回 K 個點。
    """
    while True:
        directions = fibonacci_lattice(8 * K)
        x, y, z = directions.T

        # 篩選出 xyz 都大於 0 的點
        mask = (x > 0) & (y > 0) & (z > 0)
        valid_directions = directions[mask]

        # 如果候選點不足 K 個，可以增加候選點再重試
        if valid_directions.shape[0] >= K:
            break

        print(f"Generating more points..., from {K} to {int(K * 1.1)}")
        K = int(K * 1.1)  # 增加候選點數量

    # random remove points to make sure we have exactly K points
    np.random.shuffle(valid_directions)
    return valid_directions[:K]


def ray_intersects_box(direction, x_range, y_range, z_range):
    """
    檢查一條從原點出發的射線是否與長方體相交。
    direction: 單位向量，表示射線方向，形狀為 (3,)
    x_range, y_range, z_range: 長方體的範圍，例如 [x_min, x_max]
    返回: True 如果相交, False 如果不相交。
    """
    d_x, d_y, d_z = direction
    t_min = 0.0  # 射線從原點出發，t >= 0
    t_max = np.inf

    # 對每個坐標軸計算 t 的範圍
    for d, rng in [(d_x, x_range), (d_y, y_range), (d_z, z_range)]:
        if abs(d) < 1e-6:  # 如果方向分量幾乎為 0
            if rng[0] <= 0 <= rng[1]:  # 原點在範圍內，射線可能無限長
                continue
            else:
                return False
        else:
            # 計算射線與邊界的交點
            t1 = rng[0] / d
            t2 = rng[1] / d
            if d > 0:
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
            else:
                t_min = max(t_min, t2)
                t_max = min(t_max, t1)

    # 如果 t_min <= t_max 且 t_min >= 0，則射線與長方體相交
    return t_min <= t_max and t_max >= 0


def check_direction(direction, x_range, y_range, z_range):
    """
    檢查方向是否有效（不過濾完全負方向）並與長方體相交。
    返回：(是否相交, 方向)。如果不相交，方向為 None。
    """
    # 這裡可以選擇是否過濾完全負方向，為了通用性，先不過濾
    if ray_intersects_box(direction, x_range, y_range, z_range):
        return True, direction
    return False, None


def get_intersecting_rays(x_range, y_range, z_range, N, n_jobs=-1):
    """
    給定長方體範圍和所需射線數量 N, 回傳正好 N 條與長方體相交的射線方向。
    使用並行化加速篩選。
    x_range, y_range, z_range: 長方體的範圍，例如 [x_min, x_max]
    N: 所需射線數量
    n_jobs: 並行工作進程數，-1 表示使用所有可用核心
    返回：與長方體相交的射線方向列表，形狀為 (N, 3)。
    """
    # 初始生成較多的候選射線，確保能找到足夠多的相交射線
    K = N * 10  # 初始候選數量設為 N 的 10 倍
    max_attempts = 5  # 最多嘗試 10 次增加 K

    for attempt in range(max_attempts):
        # 生成 Fibonacci lattice 上的方向
        directions = fibonacci_lattice_positive_vectorized(K)

        # 使用並行化篩選與長方體相交的射線
        delayed_check = delayed(check_direction)
        results = Parallel(n_jobs=n_jobs)(
            delayed_check(direction, x_range, y_range, z_range)
            for direction in directions
        )

        # 提取相交的方向
        intersecting_directions = [result[1] for result in results if result[0]]

        # 檢查是否找到足夠多的相交射線
        if len(intersecting_directions) >= N:
            # 如果找到的相交射線數量 >= N，則選取前 N 條
            return np.array(intersecting_directions[:N])
        else:
            # 如果不夠，增加候選數量 K
            orig_K = K
            K = int(K * min(max(N // max(len(intersecting_directions), 1), 1.1), 100))
            print(
                f"Attempt {attempt + 1}: Found {len(intersecting_directions)} intersecting rays, less than {N}. Increasing K from {orig_K} to {K}."
            )

    # 如果多次嘗試後仍不足 N 條，拋出錯誤
    raise ValueError(
        f"Unable to find {N} intersecting rays after {max_attempts} attempts. Found only {len(intersecting_directions)} intersecting rays."
    )


params = get_intersecting_rays(EJb, ECb, ELb, num_sample)
if DRY_RUN:
    energies = [np.random.randn(max_level) for _ in range(num_sample)]
else:
    energies = [
        calculate_spectrum(flxs, EJ, EC, EL)
        for EJ, EC, EL in tqdm(params, desc="Calculating")
    ]


scq.settings.PROGRESSBAR_DISABLED = False

# we can flip the data around 0.5 to make the other half
# since the fluxonium is symmetric
flxs = np.concatenate([flxs, 1.0 - flxs[::-1]])
for i in range(len(params)):
    energies[i] = np.concatenate([energies[i], energies[i][::-1]])

params = np.array(params)
energies = np.array(energies)
Ebounds = np.array((EJb, ECb, ELb))

os.makedirs(os.path.dirname(data_path), exist_ok=True)
dump_data(data_path, flxs, params, energies, Ebounds)

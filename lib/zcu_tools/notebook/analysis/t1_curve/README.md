# `t1_curve` 模塊重點文檔

**Last updated:** 2026-07-02 — t1_curve_fit params handoff

Fluxonium T1 vs. flux 的分析工具：從實測 T1 資料反推各噪聲通道的品質因子 (Q) / 準粒子密度 (x_qp)，並與理論 T1 曲線比對作圖。

## 單位慣例

- 頻率 `freqs`: GHz；`omegas` (角頻率): rad/ns；兩者由 `freq2omega` 互轉。
- T1 / T1err: ns。
- 溫度 `Temp`: K (典型 ~20 mK)。
- 能量 `params = (EJ, EC, EL)`: GHz。
- Dipole / spectral density: GHz，Q = T1 × dipole → 無量綱。

## 檔案與角色

### `utils.py` — 基礎工具

- `freq2omega(f)`：GHz → rad/ns。
- `convert_eV_to_Hz(v)`：eV → Hz（用於超導能隙）。
- `calc_therm_ratio(omega, T)`：計算 `ℏω/(kT)`，是所有 spectral density 的核心 Bose 因子參數。
- `format_exponent(n)`：把數字渲染成 LaTeX 科學記號，供圖例用。

### `base.py` — 通用作圖 / 擬合 / 溫度搜尋

- `find_proper_Temp(guess_Temp, calc_Q_fn)`：以「Q 在各 omega 上離散度最小」為目標 (`L-BFGS-B`, 10–300 mK bounds) 擬合有效溫度。`calc_Q_fn(T)` 必須回傳 Q(ω) 陣列。
- `plot_Q_vs_omega(omegas, Q, Qerr, Qname)`：log-log errorbar 散點。
- `add_Q_fit(ax, omegas, Q, omega_range=None, fit_constant=False)`：對 log-Q vs log-ω 做線性擬合得到 `Q(ω) = Q_0 ω^a`；`fit_constant=True` 時改擬合幾何平均常數。回傳 `(omegas_used, fit_Qs)`。
- `plot_t1_vs_elements(dipoles, T1s, T1errs, ...)`：T1 vs |d₀₁|² 散點，覆蓋 ±2σ 的常數 Q 參考帶 (`product2val` 用於把 `T1·|d|²` 乘積轉成欲顯示的物理量)。
- `plot_sample_t1(...)`：T1 vs normalized flux `φ_ext/φ₀`，上軸同步顯示電流/電壓（透過 `value2flux`/`flux2value`）。實測 device-value array 進入 plotting 前會正規化成 `np.float64` ndarray，避免 scalar/array overload 影響後續 errorbar 與 limits 計算。
- `plot_t1_with_sample(s_dev_values, s_T1s, s_T1errs, flux_half, flux_period, params, t_fluxs, *, name, noise_name, noise_values, Temp, **other_noise_options)`：疊加多條理論 T1(φ) 曲線；`name` ∈ {`Q_cap`, `x_qp`, `Q_ind`}；`noise_values` 中元素可為 float 或 callable `f(ω, T)`（可變 Q(ω) 模型）；底層呼叫 `simulate.fluxonium.calculate_eff_t1_vs_flux_fast`（**收 `params` tuple、自己 eigensolve，不再收 `Fluxonium`/`spectrum_data`**）。
- `plot_eff_t1_with_sample(...)`：用法相同，但直接接受已算好的 `t1_effs`。

### `t1_curve_fit.py` / `fit.py` — 四參數 T1 noise fit

- `T1FitParams(Q_cap, x_qp, Q_ind, Temp)`：fit 初值與結果容器；`Temp` 單位 K，其餘沿用 scqubits noise option 語義。
- `fit_t1_noise_params(fluxs, T1s, params, *, init, bounds=None, fixed=(), T1errs=None, residual_mode="log", progress=False, ...)`：用 `least_squares` 一次擬合 `Q_cap`、`x_qp`、`Q_ind`、`Temp`。`fluxs` 是 normalized flux；`T1s/T1errs` 是 ns；`params=(EJ,EC,EL)` 是 GHz。`progress=True` 時用 repo 的 progress-bar backend 顯示 residual evaluation 進度。
- 擬合在 log-parameter 空間進行，所有參數必須為正；`bounds` 用參數名部分覆蓋預設範圍，`fixed` 用參數名固定任意多個參數，固定值取自 `init`。
- 預設 residual 是 log T1；若提供 `T1errs`，finite positive error 會轉成權重，`NaN` 表示該點不加權。
- `success=True` 只表示 SciPy optimizer 達到終止條件；是否可信要看 residual、`reduced_chi2`、stderr 與參數是否貼近 bounds。固定參數的 stderr 回報為 0，代表未估計而不是物理不確定度為 0。
- `t1_curve_fit.py` 是明確的 public module name；`fit.py` 保留為既有 import path，兩者 export 同一組 API。
- notebook 若要把 all-in-one fit 結果交給後續模擬，使用 `QubitParams.set_t1_curve_fit(T1CurveFit(...))` 寫入 `params.json` 的 `t1_curve_fit` section。這個 section 只放 noise params、stderr、fixed/free、bounds 與 fit metadata；sample arrays、residual arrays 與 dense model curve 留在 notebook 輸出或資料檔。

### `Qcap.py` — 介電耗散 (電容通道)

- `charge_spectral_density(ω, T, EC)`：`16·EC·coth(|ℏω/2kT|)/(1+e^{-ℏω/kT})`（對應 scqubits `t1_capacitive`）。
- `calc_cap_dipole(params, n_elements, ω, T)`：`|⟨0|n̂|1⟩|² · [S(ω)+S(-ω)]`。
- `calc_Qcap_vs_omega(params, ω, T1s, n_elements, T1errs=None, Temp=20mK)`：`Q_cap = T1 · dipole`，可選回傳誤差。

### `Qind.py` — 電感耗散

- `inductive_spectral_density(ω, T, EL)`：`2·EL·coth(|ℏω/2kT|)/(1+e^{-ℏω/kT})`。
- `calc_ind_dipole(params, phi_elements, ω, T)`：用 `|⟨0|φ̂|1⟩|²`。
- `calc_Qind_vs_omega(...)`：結構同 Qcap。

### `Qqp.py` — 準粒子穿隧 (quasiparticle)

- `DELTA_ALUMINUM = 3.4e-4` eV：鋁超導能隙預設值。
- `qp_spectral_density(ω, T, EJ, Delta_eV)`：依 Smith et al. (2020) Eq. S23/19 的 `Re Y_qp / x_qp` 公式（含 `sp_special.kv(0, ...)` 修飾 Bessel）；對齊 scqubits：`Re Y_qp` 只依 `|ω|`，但 spectral-density 前因子保留 signed `ω`，讓 `S(ω)+S(-ω)` 正確。
- `calc_qp_oper(params, flux, return_dim=4, esys=None)`：回傳 `sin(φ/2)` 運算子矩陣；⚠ 假設 flux 併入 *cosine* 項（與某些文獻不同），故透過 `α=0.5, β=πflux` 做 `φ → φ + 2π·flux` 的平移。
- `calc_qp_dipole(params, sin2_elements, ω, T, Delta_eV)`：`|⟨0|sin(φ/2)|1⟩|² · [S_qp(ω)+S_qp(-ω)]`。
- `calc_Qqp_vs_omega(...)`：回傳的「Q_qp」實際上是 `1/x_qp`，代入 `plot_t1_with_sample(name="x_qp", ...)` 時要注意此反比關係。

## 典型使用流程

1. 用 `scqubits.Fluxonium` 在 `t_fluxs` 解出 `spectrum_data`，取樣點 flux 處求 `omegas` 與矩陣元 (`n`, `φ`, 或 `sin(φ/2)`)。
2. 呼叫 `calc_Q{cap,ind,qp}_vs_omega` 得到 Q(ω) 與誤差。
3. 若想自洽找溫度：`find_proper_Temp(T0, lambda T: calc_Qxxx_vs_omega(..., Temp=T))`。
4. `plot_Q_vs_omega` + `add_Q_fit` 檢視頻率相依性。
5. `plot_t1_with_sample` 把候選 Q 值 / Q(ω) 函數疊回 T1(φ) 曲線做視覺比對。

## 注意事項

- 所有 `*_vs_omega` 的「Q」是 `T1 · dipole` 的直接乘積；若 dipole 定義改（例如用 `sin(φ/2)` vs `n̂`），同一 T1 會得到不同 Q。
- `plot_t1_with_sample` 的 `other_noise_options` 以 `**kwargs` 傳給 `calculate_eff_t1_vs_flux_fast`（fast 版接受 `i`/`j`/`cutoff`/`qub_dim` 與 per-channel 選項；**fast 版固定 `total=True`，傳 `total=False` 或任何不支援的 kwarg 會 raise `UnsupportedNoiseOptionError`（不靜默丟棄）**——要 `total=False` 得用舊 scqubits `_with` 版）。
- `Qqp` 的 `calc_qp_oper` 和其他 Q 通道不同：需要自己先算 `sin(φ/2)` 矩陣元，不能直接丟 `n_elements` / `phi_elements`。

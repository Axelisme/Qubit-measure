# `fit_tools` 模塊重點文檔

**Last updated:** 2026-07-22 — nearest-branch f01 flux correction

`fit_tools` 放跨 T1/T2 分析都會用到的純計算工具。它不包含任何 T1/T2 物理機制模型；機制模型留在各自的 `t1_curve` / `t2_curve` 模塊。

## 檔案與角色

### `flux.py` — f01-based flux calibration

- `predict_f01_mhz(params, fluxs)`：用 fluxonium 能階計算 f01，回傳 MHz。
- `predict_domega_dflux(params, fluxs)`：用有限差分計算 `domega01/dflux`，回傳 rad/us/Phi0。
- `correct_flux_from_f01(dev_values, f01_freqs, params, flux_half, flux_period, ...)`：用實測 f01 校正由 device value 換算得到的 normalized flux；`f01_freqs` 單位是 GHz。校正候選會先展開 periodic / mirror equivalent flux branch，再選離 raw flux 最近者，避免 half-flux 附近的點從 0.49 直接跳到等價但較遠的 0.51。
- `choose_current_scale_from_f01(...)`：在候選電流縮放間用 f01 residual 選擇較一致的單位，回傳最佳 scale 與 report table。

### `weights.py` — residual weighting

- `MeasurementErrorPolicy` 描述未知 measurement error 的填補方式與誤差下限。
- `FluxResidualWeighting` 描述 residual 在 flux 軸上的權重方式。
- `build_flux_residual_weights(...)` 產生 per-sample residual multiplier；`mode="equal_flux_bin"` 時，同一 flux bin 內每個點乘上 `1/sqrt(N_bin)`，讓每個 occupied flux bin 對線性 least-squares loss 的總權重相同。
- `resolve_measurement_errors(...)` 將 `NaN` error 依 policy 補成 finite effective error。`nan_policy="bin_median"` 先用同一 flux bin 的 finite error median，沒有時退回全域 median。

### `loss.py` — least-squares diagnostics

- `least_squares_cost(residuals)` 回傳 `0.5 * sum(residuals**2)`。
- `reduced_chi2_from_cost(cost, observation_count, free_parameter_count)` 用 effective observation count 計算 reduced chi2；flux-bin 平衡時 observation count 是 occupied bin 數，而不是 sample 數。

## 設計原則

- 公共模組只處理資料軸、誤差與 residual 權重，不知道 `Q_cap`、`A_phi`、`n_th` 等物理參數。
- T1/T2 各自負責把物理模型轉成 residual，然後呼叫這裡的 helper 做 shared weighting。

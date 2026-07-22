# `fit_tools` 模塊重點文檔

**Last updated:** 2026-07-22 — sample merge workflow boundary

`fit_tools` 放跨 T1/T2 分析都會用到的純計算工具。它不包含任何 T1/T2 物理機制模型；機制模型留在各自的 `t1_curve` / `t2_curve` 模塊。

## 檔案與角色

### `flux.py` — f01-based flux calibration

- `predict_f01_mhz(params, fluxs)`：用 fluxonium 能階計算 f01，回傳 MHz。
- `predict_domega_dflux(params, fluxs)`：用有限差分計算 `domega01/dflux`，回傳 rad/us/Phi0。
- `correct_flux_from_f01(dev_values, f01_freqs, params, flux_half, flux_period, ...)`：用實測 f01 校正由 device value 換算得到的 normalized flux；`f01_freqs` 單位是 GHz。校正候選會先展開 periodic / mirror equivalent flux branch，再選離 raw flux 最近者，避免 half-flux 附近的點從 0.49 直接跳到等價但較遠的 0.51。
- `choose_current_scale_from_f01(...)`：在候選電流縮放間用 f01 residual 選擇較一致的單位，回傳最佳 scale 與 report table。

### `sample_merge.py` — samples.csv canonicalization

- `FluxFrame.from_result_dir(result_dir)` 從 `params.json` 的 `fluxdep_fit` 讀出 `(EJ, EC, EL)`、`flux_half` 與 `flux_period`，作為一個 source/target flux frame。
- `SampleSource(path, unit, source_result_dir=..., current_scale_to_source_frame=..., fit_batch_flux_offset=...)` 描述一個原始 sample 檔。`unit` 是報告標籤；實際數值轉換只由 `current_scale_to_source_frame` 控制，避免把 source frame 本來就是 A 的資料再次乘成 mA。
- `merge_sample_sources(target_result_dir, sources)` 先把每個 source 的 device value 轉成 source normalized flux，再映射到 target frame 的 canonical `calibrated mA` 與 `Flux`；可選 `fit_batch_flux_offset=True` 對該 source 用 f01 擬合單一 batch `delta_flux`。`batch_flux_offset_reference="source"` 使用 source model，是既有語義；`"target"` 使用 target model，對應診斷圖上的 target `measured-model` residual。`batch_flux_offset_objective` 可選 `soft_l1`、`median_abs`、`mean_abs`、`rms`；sample3 這種 shape mismatch 明顯的 batch 需要在 report 中比較不同 objective 的結果。輸出的 `merged` 表只保留 T1/T2 workflow 需要的欄位；source label、unit、batch offset 與 residual 放在 `diagnostics` / report。
- `write_merged_samples(...)` / `write_sample_merge_report(...)` 分別寫乾淨的 `samples.csv` 與診斷 report；`plot_sample_merge_f01_diagnostics(...)` 用 target model 檢查合併後 f01 residual。

### `weights.py` — residual weighting

- `MeasurementErrorPolicy` 描述未知 measurement error 的填補方式與誤差下限。
- `FluxResidualWeighting` 描述 residual 在 flux 軸上的權重方式。
- `build_flux_residual_weights(...)` 產生 per-sample residual multiplier；`mode="equal_flux_bin"` 時，同一 flux bin 內每個點乘上 `1/sqrt(N_bin)`，讓每個 occupied flux bin 對線性 least-squares loss 的總權重相同。
- `resolve_measurement_errors(...)` 將 `NaN` error 依 policy 補成 finite effective error。`nan_policy="bin_median"` 先用同一 flux bin 的 finite error median，沒有時退回全域 median。

### `loss.py` — least-squares diagnostics

- `least_squares_cost(residuals)` 回傳 `0.5 * sum(residuals**2)`。
- `reduced_chi2_from_cost(cost, observation_count, free_parameter_count)` 用 effective observation count 計算 reduced chi2；flux-bin 平衡時 observation count 是 occupied bin 數，而不是 sample 數。

## 設計原則

- 公共模組只處理資料軸、sample canonicalization、誤差與 residual 權重，不知道 `Q_cap`、`A_phi`、`n_th` 等物理參數。
- T1/T2 各自負責把物理模型轉成 residual，然後呼叫這裡的 helper 做 shared weighting。
- `samples.csv` 合併與 source frame/unit 修正屬於 `sample_merge.py`；`T1_curve.md` / `T2_curve.md` 只讀已 canonicalized 的 sample 表，再做 analysis-window 內的小範圍 point-wise f01 correction。

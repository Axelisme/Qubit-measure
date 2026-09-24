# `fit_tools` 模塊重點文檔

**Last updated:** 2026-08-18 — flux-first v2 SampleMerge

`fit_tools` 放跨 T1/T2 分析都會用到的純計算工具。它不包含任何 T1/T2 物理機制模型；機制模型留在各自的 `t1_curve` / `t2_curve` 模塊。

## 檔案與角色

### `flux.py` — normalized-flux f01 correction

- `predict_f01_mhz(params, fluxs)`：用 fluxonium 能階計算 f01，回傳 MHz。
- `predict_domega_dflux(params, fluxs)`：用有限差分計算 `domega01/dflux`，回傳 rad/us/Phi0。
- `correct_flux_from_f01(raw_fluxs, f01_freqs_ghz, params, *, max_abs_flux_correction=0.03)`：以實測 f01 校正 dimensionless normalized flux；`f01_freqs_ghz` 明確以 GHz 輸入，內部先轉成 MHz 再呼叫 MHz-based predictor。校正候選會先展開 periodic / mirror equivalent flux branch，再選離 raw flux 最近者；tie 規則 deterministic（等距時 periodic 獲勝）。超過 `max_abs_flux_correction` 的行拒絕並保留 raw flux。回傳 `F01FluxCorrectionResult` 只有 raw/corrected flux、accepted mask 與 applied correction —— 不再回傳需要單一 device frame 才定義的 `corrected_dev_values`。explicit flux 視為已校準，由 caller 決定是否套用此校正。
- `align_flux_to_window(fluxs, analysis_flux_range)`：analysis-local integer-period alignment。window 必須 finite、strictly increasing 且 width `<= 1.0`；每一 flux 選離 window midpoint 最近的 `flux + k`（`k ∈ ℤ`）branch，等距 tie 選較小的 resulting flux。回傳 aligned flux、integer shift 與 alignment 後的 in-window mask；width 超過一個 period 直接 fail。

### `sample_merge.py` — flux-first v2 SampleMerge

- `FluxFrame(params, dev_unit, flux_int, flux_period, label)`：analysis affine flux frame。construction 驗證 A/V `dev_unit`、finite `flux_int` 與 positive finite `flux_period`，f01 model `params` 必須是三個 finite float；`flux_from_dev_value()` / `dev_value_from_flux()` 提供 scalar/array round trip（`(dev_value - flux_int) / flux_period`）。`from_result_dir(result_dir, *, dev_unit, label=None)` 從 `params.json` 的 `fluxdep_fit` 讀 `(EJ, EC, EL)` 與 `flux_int` / `flux_period`；`dev_unit` 必填，因為 persisted params 沒有 unit metadata。
- `SampleSource(path, label=..., fallback_frame=..., integer_flux_offset=0, fit_batch_flux_offset=..., batch_flux_offset_objective=..., batch_flux_offset_range=..., max_abs_batch_flux_offset=..., f01_fit_scale_mhz=...)` 只接受 v2 CSV（`validate_sample_table_v2`）。`fallback_frame` 供缺 row frame 的 migrated rows 解析 flux；`integer_flux_offset` 是 caller-declared integer branch 對齊，merge 不猜測 unit 或 integer branch。branch 對齊後的唯一非整數調整是 bounded batch fit（`max_abs_batch_flux_offset` 限定）。`unit`、`source_result_dir`、`current_scale_to_source_frame` 與 legacy `calibrated mA` / `Flux` 相容已移除。
- `merge_sample_sources(*, target_frame, sources)`：唯一 authoritative target identity。pipeline：`resolve_sample_flux`（explicit → row frame → fallback frame，共用 provenance SSOT）→ caller `integer_flux_offset` → optional small batch offset（對 target f01 model 擬合單一 `delta_flux`；`batch_flux_offset_objective` 可選 `soft_l1`、`median_abs`、`mean_abs`、`rms`）→ target frame `dev_value`。輸出完整 flat v2 coordinate：adjusted `flux` 與 target `dev_value` / `dev_unit` / `flux_int` / `flux_period`，加上 caller-owned measurement columns（不 alias-canonicalize）。diagnostics 逐 row 記錄 resolution `sources` provenance，並明示 integer offset、batch correction 與 target frame；A-source 與 V-source 可合併到同一 target frame，不直接比較 raw device values。unresolved rows / legacy tables 在 mutation 前 fast-fail。
- `write_merged_samples(...)` / `write_sample_merge_report(...)` 分別寫乾淨的 `samples.csv` 與診斷 report；兩者都拒絕覆寫任何 merge source CSV。`plot_sample_merge_f01_diagnostics(...)` 用 target model 檢查合併後 f01 residual。

### `weights.py` — residual weighting

- `MeasurementErrorPolicy` 描述未知 measurement error 的填補方式與誤差下限。
- `FluxResidualWeighting` 描述 residual 在 flux 軸上的權重方式。
- `build_flux_residual_weights(...)` 產生 per-sample residual multiplier；`mode="equal_flux_bin"` 時，同一 flux bin 內每個點乘上 `1/sqrt(N_bin)`，讓每個 occupied flux bin 對線性 least-squares loss 的總權重相同。
- `resolve_measurement_errors(...)` 將 `NaN` error 依 policy 補成 finite effective error。`nan_policy="bin_median"` 先用同一 flux bin 的 finite error median，沒有時退回全域 median。

### `loss.py` — least-squares diagnostics

- `least_squares_cost(residuals)` 回傳 `0.5 * sum(residuals**2)`。
- `reduced_chi2_from_cost(cost, observation_count, free_parameter_count)` 用 effective observation count 計算 reduced chi2；flux-bin 平衡時 observation count 是 occupied bin 數，而不是 sample 數。

### `effective_temperature.py` — attenuator-chain thermal noise

- `ThermalAttenuatorStage(name, Temp_K, attenuation_db)` 描述一個實際位於該溫度的 passive attenuator；任意多層用 tuple/list 串起來。
- `calculate_thermal_chain(frequencies_hz, stages, input_temperature_K=300.0, impedance_ohm=50.0)` 用 passive attenuator cascade 計算 output-referred PSD、等效 photon number 與等效溫度。每層視為無長度 lumped attenuator，只輸入溫度與 attenuation；自身熱噪聲使用 `linear_loss = 10^(att_db/10)` 對應的 emissivity `1 - 1/linear_loss`，source 與各層 emission 再乘以下游 attenuation 後相加。
- `evaluate_thermal_chain_at_frequency(...)` 對單一 probe/readout 頻率回傳 `T_eff`、`n` 與 PSD，避免 notebook 先插值再反解。
- `plot_thermal_chain_psd(...)` 與 `plot_effective_temperature_vs_frequency(...)` 回傳 `(fig, ax)`，notebook 只決定頻率軸、probe frequency、highlight range 與存檔路徑。PSD 圖維持舊 notebook 的顯示口徑：各溫區曲線是 raw source PSD，黑色 `Effective` 曲線才是 attenuator chain 後的 output-referred total。

## 設計原則

- 公共模組只處理資料軸、sample canonicalization、誤差與 residual 權重，不知道 `Q_cap`、`A_phi`、`n_th` 等物理參數。
- T1/T2 各自負責把物理模型轉成 residual，然後呼叫這裡的 helper 做 shared weighting。
- `samples.csv` 合併與 target-frame 映射屬於 `sample_merge.py`；`T1_curve.md` / `T2_curve.md` 只讀已 canonicalized 的 sample 表，再做 analysis-window 內的小範圍 point-wise f01 correction。

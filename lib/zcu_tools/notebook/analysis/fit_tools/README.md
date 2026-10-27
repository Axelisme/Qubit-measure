# `fit_tools` 模塊重點文檔

**Last updated:** 2026-10-27

**讀者：** lab-author。T1／T2 共用的座標校正、樣本合併與 residual 工具。

`fit_tools` 放跨 T1/T2 分析都會用到的純計算工具。T1/T2 物理機制模型由各自的 `t1_curve` / `t2_curve` 模塊擁有。

## 檔案與角色

### `flux.py` — normalized-flux f01 correction

- `predict_f01_mhz(params, fluxs)`：用 fluxonium 能階計算 f01，回傳 MHz。
- `predict_domega_dflux(params, fluxs)`：用有限差分計算 `domega01/dflux`，回傳 rad/us/Phi0。
- `correct_flux_from_f01(raw_fluxs, f01_freqs_ghz, params, *, max_abs_flux_correction=0.03)`：以實測 f01 校正 dimensionless normalized flux；`f01_freqs_ghz` 明確以 GHz 輸入，內部先轉成 MHz 再呼叫 MHz-based predictor。校正候選會先展開 periodic / mirror equivalent flux branch，再選離 raw flux 最近者；tie 規則 deterministic（等距時 periodic 獲勝）。超過 `max_abs_flux_correction` 的行拒絕並保留 raw flux。回傳 `F01FluxCorrectionResult` 只有 raw/corrected flux、accepted mask 與 applied correction。作者決定是否在 load stage 前呼叫。整欄頻率缺少傳 None，逐值缺少傳 NaN，工具以作者 flux 的 model f01 補值。校正結果只描述本工具的操作，不由 analysis stage 推定或重建。
- `align_flux_to_window(fluxs, analysis_flux_range)`：analysis-local integer-period alignment。window 必須 finite、strictly increasing 且 width `<= 1.0`；每一 flux 選離 window midpoint 最近的 `flux + k`（`k ∈ ℤ`）branch，等距 tie 選較小的 resulting flux。回傳 aligned flux、integer shift 與 alignment 後的 in-window mask；width 超過一個 period 直接 fail。

### `sample_merge.py` — record flattening and concatenation

`merge_sample_sources(sources=[path1, path2])` 只匯整 record JSONL。它使用 resources 的 `JsonlRecordTable.read(flatten=True)`，每個 run_id 取最後一整行。巢狀物件攤成點號欄名，例如 `devices.bias.value`，再依來源順序串接。來源值保留，不解釋舊欄位別名。缺量測、裝置或 frame 欄位、未用 NaN 與額外 metadata 都不阻擋匯整。

回傳 `SampleMergeResult`。`merged` 是攤平後的 DataFrame，`source_paths` 保存所有來源的絕對路徑，包含空檔。`write_merged_samples(result, path)` 將表寫成 CSV，建立缺少的父目錄，並拒絕覆寫來源 JSONL。它可以替換既有的輸出 CSV。

匯整不算 flux、不收 fallback frame，也不做 integer offset、batch f01 fit 或 target-frame 映射。作者另外呼叫 resources 的 [`flux_from_frame`](../../../resources/record_table/README.md#作者-flux-換算)，或自行建立 flux 陣列，再決定是否呼叫 f01 校正。Notebook 組裝示例見 [Sample_merge](../../../../../notebook_md/analysis/Sample_merge.md)。

### `flux_presentation.py` — author-selected device axis

`add_device_axis(ax, frame, device_name=..., values=...)` 在 normalized-flux 橫軸圖加上方儀器值軸。
`frame` 接受 `analysis.flux.FluxFrame`，或明確的單 device `resources.record_table.RecordFluxFrame`。
多 device 時，`values` 必須提供其他 frame device 的當下儀器值。
函式呼叫 `analysis.flux.actual_flux_points` 算出選定 device 當下的 integer，再以其 signed period 建立正反轉換。
缺其他 device 的值會報錯，不假設它停在 integer。
單 device 可以省略 `values`。
單／多 device 的 `device_name` 都共用 `cfg_model.validate_device_name`。
它拒絕 ASCII 點號、空字串與全空白，有效名稱原樣用於查找與軸標。

第二軸使用選定 device 宣告的原生單位，不換成 milli 單位。
回傳的 Matplotlib SecondaryAxis 隨主軸 limits 更新，也支持負 period。
作者選擇 device、frame 與當下值；不加第二軸時，T1/T2 繪圖只需要 normalized flux。
通用 flux 數學的型別與契約見 [analysis README](../../../analysis/README.md#多-device-flux-座標)。

### `weights.py` — residual weighting

- `MeasurementErrorPolicy` 描述未知 measurement error 的填補方式與誤差下限。
- `FluxResidualWeighting` 描述 residual 在 flux 軸上的權重方式。
- `build_flux_residual_weights(...)` 產生 per-sample residual multiplier；`mode="equal_flux_bin"` 時，同一 flux bin 內每個點乘上 `1/sqrt(N_bin)`，讓每個 occupied flux bin 對線性 least-squares loss 的總權重相同。
- `resolve_measurement_errors(...)` 將 `NaN` error 依 policy 補成 finite effective error。`nan_policy="bin_median"` 先用同一 flux bin 的 finite error median，沒有時退回全域 median。

### `loss.py` — least-squares diagnostics

- `least_squares_cost(residuals)` 回傳 `0.5 * sum(residuals**2)`。
- `reduced_chi2_from_cost(cost, observation_count, free_parameter_count)` 用 effective observation count 計算 reduced chi2；flux-bin 平衡時 observation count 是 occupied bin 數，而不是 sample 數。

### Identifiability diagnostics

`identifiability.py` 擁有 `FitCorrelation` 與 `WeakParameter`。
`calculate_fit_correlation(...)` 保留按自由參數順序排列的完整 natural-log covariance。
它以 column-normalized Jacobian 的 SVD 計算 covariance，再轉回 natural-log 參數座標，並乘上 optimizer-cost reduced χ²。
Rank 與被截掉方向的判定使用同一個相對容差：float64 epsilon 乘以 Jacobian 的最大維度。
這避免先形成 JᵀJ 而平方條件數，也保留欄縮放下的 rank。
Correlation 由未縮放矩陣正規化，cost 為 0 時仍保留相關資訊。
零欄或在被截掉方向上有分量的參數，其 covariance 對角為 inf，correlation 列與欄為 NaN，fit stderr 為 inf。
很小但獨立的非零欄仍有有限且可能很大的 variance，不因幅度小而截掉。
秩不足時條件數為 inf；Jacobian 或 cost 非有限時回傳 None，fit 的自由參數 stderr 為 inf。

`find_weak_parameters(...)` 擁有共用規則：|correlation| > 0.95 或 σ(ln p) > 1。
`CORRELATION_THRESHOLD` 與 `LOG_STD_THRESHOLD` 定義這兩個門檻。
函式回傳每個弱參數及全部理由，理由包含觀察值。σ(ln p) = inf 仍依第二條規則標弱，並說明資料不約束此參數。沒有矩陣時回傳空 tuple。
`identifiability_presentation.py` 的 `plot_fit_correlation(...)` 回傳標數值的熱圖，圖上標明 covariance 位於 log 參數空間。
`make_weak_parameter_table(...)` 回傳參數與原因表。T1/T2 notebook 直接呼叫這些共用函式。

### `effective_temperature.py` — attenuator-chain thermal noise

- `ThermalAttenuatorStage(name, Temp_K, attenuation_db)` 描述一個實際位於該溫度的 passive attenuator；任意多層用 tuple/list 串起來。
- `calculate_thermal_chain(frequencies_hz, stages, input_temperature_K=300.0, impedance_ohm=50.0)` 用 passive attenuator cascade 計算 output-referred PSD、等效 photon number 與等效溫度。每層視為無長度 lumped attenuator，只輸入溫度與 attenuation；自身熱噪聲使用 `linear_loss = 10^(att_db/10)` 對應的 emissivity `1 - 1/linear_loss`，source 與各層 emission 再乘以下游 attenuation 後相加。
- `evaluate_thermal_chain_at_frequency(...)` 對單一 probe/readout 頻率回傳 `T_eff`、`n` 與 PSD，避免 notebook 先插值再反解。
- `plot_thermal_chain_psd(...)` 與 `plot_effective_temperature_vs_frequency(...)` 回傳 `(fig, ax)`，notebook 只決定頻率軸、probe frequency、highlight range 與存檔路徑。PSD 圖維持舊 notebook 的顯示口徑：各溫區曲線是 raw source PSD，黑色 `Effective` 曲線才是 attenuator chain 後的 output-referred total。

## 設計原則

- 公共模組處理資料軸、sample canonicalization、誤差與 residual 權重。`Q_cap`、`A_phi`、`n_th` 等物理參數由 T1/T2 模型解讀。
- T1/T2 各自負責把物理模型轉成 residual，然後呼叫這裡的 helper 做 shared weighting。
- `sample_merge.py` 只匯整 JSONL 成 CSV。[`T1_curve.md`](../../../../../notebook_md/analysis/T1_curve.md) 與 [`T2_curve.md`](../../../../../notebook_md/analysis/T2_curve.md) 由作者組裝 CSV 讀取、flux 換算與選用 f01 校正。Load stage 接受逐列對應的作者 flux。Flux 換算與 f01 校正不隱藏在匯整或 loader 裡。

SampleMerge 的 CSV 是分析輸入，不是 adapter 原始記錄。匯整只攤平欄名，不改寫來源 JSONL 或 Run snapshot。

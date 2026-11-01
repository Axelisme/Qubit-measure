# Heralded T1

**讀者：** operator、agent。使用 GUI 執行單 qubit heralded T1，分析 g 類條件平均並保存各類總和。

本頁給已校準單 qubit π pulse 與 pulse readout classifier 的操作員。實驗先做 herald readout，再做 π → delay → signal readout。取數保存 g、e、other 各類的 count 與 IQ 總和，即時圖與擬合只取 g 類的 signal 條件平均。這是 host post-selection，沒有 active reset，也不檢查其他 qubit。

## 執行前

- Pi Pulse 選 `$qubit.module.x180` 的校準脈衝。
- Readout 選 `$resonator.module.readout` 的單音 pulse readout。兩次讀出共用此設定。Mux 不適用。
- Readout 必須包含 GE Post circle classifier，具有 g/e centers 與 radius。Linear 與 centroid3 不適用。座標是未扣 offset 的 normalized ADC IQ。
- Circle 將 herald 分為 g、e、other。分析只取 g 類的 signal 條件平均。
- 模擬讀出仍受 dispersive validity guard 限制。Gain 安全範圍取決於 carrier、pulse length、flux 與 truncation，不能直接套用別條讀出線的界限。

## Run 與 Analyze

Delay 使用 us，只接受 Start/Stop/Expts 時間窗。Sampling 選 linear 時掃描線性延遲；選 decay 時使用共用 T1 衰減段取樣。硬體量化後重疊的點會報錯，不會減點。

Reps 是每點每 round 的 shots，Rounds 是軟體 rounds。Relax delay 是每 shot 結束後的等待。序列依序為 herald readout、π pulse、delay、signal readout。沒有額外的 reset pulse。

Run 的 measurement 顯示通過 shot 的平均 IQ magnitude。跨 round 依通過數加權，不平均各 round 的平均值。沒有 shot 通過的點為 NaN，通過比例為 0；取消前未完成的點保留 NaN。

Analyze 取 herald-g 的 signal 條件平均 IQ。Skip leading points 的 `skip` 先從原網格移除前幾點，之後排除非有限 time／IQ，再做 IQ rotation，取實部擬合。本實驗直接呼叫 `fit_decay` 做單 exponential fit；開啟 Dual exponential 的 `dual_exp` 時改用 `fit_dual_decay`。本實驗在 `fit` 圖畫出旋轉後的資料與擬合曲線，標題顯示壽命與標準誤。Pass fraction 不當作 fit weight。分析不需要 saved cfg，有效點不足時分析會失敗。

Analyze 回傳 `HeraldingT1Analysis`：

- `t1`、`t1_err` 是主要衰減時間與標準誤，單位 us，皆不為 None。
- `t1b`、`t1b_err` 是第二個衰減時間與標準誤，單位 us；單 exponential fit 時兩者為 None。
- `fit_quality["fit"]` 保存無因次的殘差與相對誤差診斷。型別允許未提供診斷時 `fit_quality` 為 None，本實驗的 Analyze 每次成功都提供診斷。

Apply 只在操作員確認後寫回 `$qubit.t1`。

## Save 與載回

Save 將 `sums` 寫入 native `data.h5`，形狀是 (Nt, 3, 3)。後兩軸是 herald_states（g、e、other）與 components（count、ΣI、ΣQ）。時間在檔案中使用 seconds，載回 Result 時恢復 us。通過比例只在分析或即時呈現時以 g 計數除以總計數，不另存變數。舊格式缺少 `sums` 時明確報錯。Cfg 保存這次 readout 的 classifier。用本實驗載回資料後可重新 Analyze，不需要目前的 classifier。

Pass fraction 是 classifier 接受的 shot 比例，不是 ground-state purity 或 reset fidelity。FTF joint-state IQ cluster 不能當作單 qubit g/e 校準；離線走通流程也不代表物理校準完成。

# Heralded T1

本頁給已校準單 qubit π pulse 與 pulse readout classifier 的操作員。實驗先讀出一次，只保留判為 ground 的 shot，再做 π → delay → signal readout。這是 host post-selection，沒有 active reset，也不檢查其他 qubit。

## 執行前

- Pi Pulse 選 `$qubit.module.x180` 的校準脈衝。
- Readout 選 `$resonator.module.readout` 的單音 pulse readout。兩次讀出共用此設定。Mux 不適用。
- Classifier 必須在 Readout 內。Linear 使用 angle 與 threshold，angle 單位為 radians。Circle 使用 g/e centers 與 radius，缺 radius 時先完成 GE Post。座標是未扣 offset 的 normalized ADC IQ。
- Circle 只接受 g 圓內且比 e 更近的 shot。Linear 接受 ground 一側。Core 不重做分類或計數加權。
- 模擬讀出仍受 dispersive validity guard 限制。Gain 安全範圍取決於 carrier、pulse length、flux 與 truncation，不能直接套用別條讀出線的界限。

## Run 與 Analyze

Delay 使用 us。Uniform 開啟時掃描線性延遲；關閉時使用共用 T1 非均勻取樣。硬體量化後重疊的點會報錯，不會減點。

Reps 是每點每 round 的 shots，Rounds 是軟體 rounds。Relax delay 是每 shot 結束後的等待。序列依序為 herald readout、π pulse、delay、signal readout。沒有額外的 reset pulse。

Run 的 measurement 顯示通過 shot 的平均 IQ magnitude。跨 round 依通過數加權，不平均各 round 的平均值。沒有 shot 通過的點為 NaN，pass_fraction 為 0；取消前未完成的點保留 NaN。

Analyze 沿用普通 T1 的 IQ rotation 與單／雙 exponential fit。Skip leading points 先從原網格移除前幾點，之後再排除非有限 time/IQ，才做 rotation 與 fit。Pass fraction 不當作 fit weight。有效點不足時分析會失敗。Apply 只在操作員確認後寫回 `$qubit.t1`。

## Save 與載回

Save 將 `signals` 與同一 time grid 的 `pass_fraction` 放進同一個 `data.h5`，時間在檔案中使用 seconds，載回 Result 時恢復 us。Cfg 保存這次 readout 的 classifier。用本實驗載回資料後可重新 Analyze，不需要目前的 classifier。

Pass fraction 是 classifier 接受的 shot 比例，不是 ground-state purity 或 reset fidelity。FTF joint-state IQ cluster 不能當作單 qubit g/e 校準；離線走通流程也不代表物理校準完成。

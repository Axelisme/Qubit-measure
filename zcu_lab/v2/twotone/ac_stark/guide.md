# AC Stark spectroscopy

**讀者：** operator、agent。從 New Tab ▾ 選擇 `twotone` → `ac_stark`。此實驗掃描 resonator Stark tone 的 gain 與 qubit probe 的 frequency，量測平均 complex IQ。此結果用於頻譜與 Stark shift 分析。

**Last updated:** 2026-10-21

## 設定與量測

先完成 readout 與 qubit spectroscopy 校準。確認 `qubit`、`resonator` aliases 指向正確元件，並檢查 wiring、readout 與 reset。Open／Reset 從固定來源建立初值。之後若變更 alias，按 Reset 才重新取初值。Run 使用畫面上的設定。

- Stark resonator tone 使用 `res_probe` 範本，頻率取 `$resonator.freq`，起始長度為 1 us，與預設 qubit probe 同長。
- Qubit spectroscopy probe 使用 `qub_probe` 範本，起始長度為 1 us。
- Probe frequency 以 `$qubit.freq` 為中心，半寬為 `1.5 * $qubit.kappa`，共 201 點。缺校準時欄位保持未填，請手動設定。
- Stark gain 初始範圍為 0 到 0.1，共 21 點。掃描在 gain 平方上等距取點，再做硬體量化。
- Reset 可停用。Readout 優先使用 `$qubit.module.readout`，沒有時提供可編輯範本。
- Reps／Rounds 控制平均次數。Early-stop SNR 可留空，或讓每條頻譜達到指定 SNR 後提早結束平均。

每點先 reset，再啟動 Stark tone，於 Stark tone 持續期間執行 qubit probe，最後 readout。請檢查兩個 pulse 的 pre/post delay 與長度，確認所需的重疊及 readout 時間。1 us 是可編輯的初值，硬體安全仍須依功率與時序審核。若加長 Stark tone，必須同步檢查 probe 與 readout 時序，避免同一 generator 的 pulse 衝突。真實硬體 Run 前必須審核功率、接線及時序。

檢查表單設定，再按 Run。Measurement 圖顯示每條頻譜正規化後的訊號與 current trace。Reset 會重新捕捉來源並以初值取代手改值。

## 分析與寫回

Analyze 從本次 Run／Load 的固定來源讀取 `$qubit.ext.chi` 與 `$resonator.kappa`，單位為 MHz。缺值時分析回報錯誤；請使用具備必要校準來源的資料。Polynomial degree 可選 1 或 2，對 gain 平方擬合頻率。Gain cutoff 留空時使用全部 gain 資料；設定時排除大於等於該 gain 的資料。

Fit 圖顯示抽出的 resonance 與 Stark shift 曲線。Summary 的 `ac_coeff` 滿足 mean photon number = ac_coeff * gain²。請確認頻譜峰值與擬合範圍合理，再於 writeback preview 選擇 Apply，寫入 `$qubit.ext.ac_stark_coeff`。校準只在 Apply 後更新。

## 保存與載入

Save 保存本次量測設定、gain／frequency 軸與二維 complex IQ。記憶體頻率為 MHz，原生檔案頻率為 Hz。Save／Load 使用原生格式與 tag `twotone/ge/ac_stark`；請選擇相符格式的檔案。Load 保留目前表單輸入；後續 Analyze／Save 只使用載入結果及其來源。

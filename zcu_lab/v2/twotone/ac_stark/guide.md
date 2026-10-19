# AC Stark spectroscopy

從 New Tab ▾ 選擇 `twotone` → `ac_stark`。此實驗掃描 resonator Stark tone 的 gain 與 qubit probe 的 frequency，量測平均 complex IQ，不做 singleshot 分類。

## 設定與量測

先完成 readout 與 qubit spectroscopy 校準。確認 `qubit`、`resonator` aliases 指向正確元件，並檢查 wiring、readout 與 reset。Open／Reset 從固定來源建立初值，之後改 alias 不會更新已開啟的表單。Run 使用畫面上的設定。

- Stark resonator tone 使用 `res_probe` 範本，頻率取 `$resonator.freq`，起始長度為 1 us，與預設 qubit probe 同長。
- Qubit spectroscopy probe 使用 `qub_probe` 範本，起始長度為 1 us。
- Probe frequency 以 `$qubit.freq` 為中心，半寬為 `1.5 * $qubit.kappa`，共 201 點。缺校準時欄位保持未填，請手動設定。
- Stark gain 初始範圍為 0 到 0.1，共 21 點。Core 在 gain 平方上等距取點，再做硬體量化，並非 gain 等距掃描。
- Reset 可停用。Readout 優先使用 `$qubit.module.readout`，沒有時提供可編輯範本。
- Reps／Rounds 控制平均次數。Early-stop SNR 可留空，或讓每條頻譜達到指定 SNR 後提早結束平均。

每點依序 reset、非阻塞啟動 Stark tone、執行 qubit probe，再 readout。請檢查兩個 pulse 的 pre/post delay 與長度，確認所需的重疊及 readout 時間。1 us 只是初值，不是硬體安全限制。若加長 Stark tone，必須同步檢查 probe 與 readout 時序，避免同一 generator 的 pulse 衝突。真實硬體 Run 前必須審核功率、接線及時序。

檢查表單設定，再按 Run。Measurement 圖顯示每條頻譜正規化後的訊號與 current trace。Reset 會重新捕捉來源並重建初值，不保留手改值。

## 分析與寫回

Analyze 從本次 Run／Load 的固定來源讀取 `$qubit.ext.chi` 與 `$resonator.kappa`，單位為 MHz。缺值會明確失敗，不讀目前全域校準補值。Polynomial degree 可選 1 或 2，對 gain 平方擬合頻率。Gain cutoff 留空代表不裁切；設定時排除大於等於該 gain 的資料。

Fit 圖顯示抽出的 resonance 與 Stark shift 曲線。Summary 的 `ac_coeff` 滿足 mean photon number = ac_coeff * gain²。請確認頻譜峰值與擬合範圍合理，再於 writeback preview 選擇 Apply，寫入 `$qubit.ext.ac_stark_coeff`。Analyze 不自動套用校準。

## 保存與載入

Save 保存本次 resolved cfg、gain／frequency 軸與二維 complex IQ。記憶體頻率為 MHz，原生檔案頻率為 Hz。GUI 沿用 core 的 canonical 格式與 tag `twotone/ge/ac_stark`，不轉換舊格式。Load 不覆寫目前表單；後續 Analyze／Save 使用載入結果，不用表單補資料。

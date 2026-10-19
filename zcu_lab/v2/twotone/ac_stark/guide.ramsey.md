# AC Stark Ramsey

從 New Tab ▾ 選擇 `twotone` → `ac_stark_ramsey`。此實驗量測固定長度 Stark tone 下的 Ramsey delay fringes，並用 FFT 顯示其頻率隨 Stark gain 的變化。

## 設定與量測

先完成 readout、qubit frequency 與 pi/2 pulse 校準。確認 `qubit`、`resonator` aliases、接線及各 module。Open／Reset 捕捉來源建立初值，Run 不重新讀取 library 或目前 alias。

- Stark resonator tone 使用 `res_probe` 範本，頻率取 `$resonator.freq`，起始長度為 10 us。這個長度不隨 Ramsey delay 掃描。
- Ramsey pi/2 pulse 優先使用 `$qubit.module.x90`，沒有時使用可編輯的 `pi2_pulse` 範本。請確認其 gain、frequency 與 waveform。
- Stark gain 初始範圍為 0 到 0.1，共 21 點。Core 在 gain 平方上等距取點，再做硬體量化。
- Ramsey delay 初始範圍為 0 到 1 us，共 101 點。Tone settling delay 初值為 0.1 us。
- Phase-ramp detune 初值為 1 MHz。第二個 pi/2 pulse 的相位增加 360 * detune * Ramsey delay，MHz 與 us 相乘。
- Reset 可停用。Readout 優先使用 `$qubit.module.readout`，缺少時使用可編輯範本。Reps／Rounds 控制平均次數。

每點先 reset，再同時開始兩條 branch。一條執行 Stark tone，另一條依序等待 settling delay、pi/2、掃描 delay、pi/2。Join 等兩條 branch 都完成後才 readout。請設定 Stark 長度，使它覆蓋所需的 Ramsey 區段，並檢查 pulse 的 pre/post delay。10 us 只是初值，不是安全限制或自動補齊的時間。

核對表單設定，再按 Run。Measurement 圖顯示平均 IQ 的共同旋轉投影及 current trace。Reset 重新捕捉來源，不保留手改值。真實硬體 Run 前仍需審核功率及時序。

## FFT 分析

Analyze 的 FFT detune 預設使用本次 acquired cfg 的 Phase-ramp detune，沒有 cfg 時用 0 MHz。此值可另行修改，只影響分析。Core 保留其附近 ±5 MHz 的 FFT 頻率窗，並從顯示頻率軸扣除此值。Gain cutoff 設定時排除大於等於該 gain 的資料。

Fit 圖上方為 Ramsey contrast map，下方為每條 trace 分別正規化的 FFT magnitude，橫軸為 gain²。FFT 圖不是 fitted Stark coefficient 或 T2，也不提供校準 writeback。請保留足夠 delay 點數與時間跨度，避免把頻率解析度或 alias 當成物理變化。

## 保存與載入

Save 保存本次 resolved cfg、gain／delay 軸與二維 complex IQ。記憶體 delay 為 us，原生檔案為 s。GUI 沿用 core 的 canonical 格式與 tag `twotone/ge/ac_stark_ramsey`。Load 不覆寫目前表單，Analyze 使用載入結果。

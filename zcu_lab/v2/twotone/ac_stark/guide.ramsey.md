# AC Stark Ramsey

**讀者：** operator、agent。從 New Tab ▾ 選擇 `twotone` → `ac_stark_ramsey`。此實驗量測固定長度 Stark tone 下的 Ramsey delay fringes，並用 FFT 顯示其頻率隨 Stark gain 的變化。

**Last updated:** 2026-10-21

## 設定與量測

先完成 readout、qubit frequency 與 pi/2 pulse 校準。確認 `qubit`、`resonator` aliases、接線及各 module。Open／Reset 捕捉來源建立初值，Run 使用表單上的設定。

- Stark resonator tone 使用 `res_probe` 範本，頻率取 `$resonator.freq`，起始長度為 10 us。這個長度在 Ramsey delay 掃描中保持固定。
- Ramsey pi/2 pulse 優先使用 `$qubit.module.x90`，沒有時使用可編輯的 `pi2_pulse` 範本。請確認其 gain、frequency 與 waveform。
- Stark gain 初始範圍為 0 到 0.1，共 21 點。掃描在 gain 平方上等距取點，再做硬體量化。
- Ramsey delay 初始範圍為 0 到 1 us，共 101 點。Tone settling delay 初值為 0.1 us。
- Phase-ramp detune 初值為 1 MHz。第二個 pi/2 pulse 的相位增加 360 * detune * Ramsey delay，MHz 與 us 相乘。
- Reset 可停用。Readout 優先使用 `$qubit.module.readout`，缺少時使用可編輯範本。Reps／Rounds 控制平均次數。

每點先 reset，再同時執行 Stark tone 與 Ramsey 序列。Ramsey 序列依序為 settling delay、pi/2、掃描 delay、pi/2。兩者都完成後才 readout。請設定 Stark 長度，使它覆蓋所需的 Ramsey 區段，並檢查 pulse 的 pre/post delay。10 us 是可編輯的固定初值；請自行設定足以覆蓋序列的安全時長。

核對表單設定，再按 Run。Measurement 圖顯示平均 IQ 的共同旋轉投影及 current trace。Reset 重新捕捉來源並以初值取代手改值。真實硬體 Run 前仍需審核功率及時序。

## FFT 分析

Analyze 的 FFT detune 預設使用本次量測設定的 Phase-ramp detune，缺少量測設定時用 0 MHz。此值可另行修改，只影響分析。分析保留其附近 ±5 MHz 的 FFT 頻率窗，並從顯示頻率軸扣除此值。Gain cutoff 設定時排除大於等於該 gain 的資料。

Fit 圖上方為 Ramsey contrast map，下方為每條 trace 分別正規化的 FFT magnitude，橫軸為 gain²。FFT 圖用於判讀頻率隨 gain² 的變化，校準值保持原樣。Stark coefficient 與 T2 須以相應的擬合實驗取得。請保留足夠 delay 點數與時間跨度，避免把頻率解析度或 alias 當成物理變化。

## 保存與載入

Save 保存本次量測設定、gain／delay 軸與二維 complex IQ。記憶體 delay 為 us，原生檔案為 s。Save／Load 使用原生格式與 tag `twotone/ge/ac_stark_ramsey`。Load 保留目前表單輸入，Analyze 使用載入結果。

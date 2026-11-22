**讀者：** operator、agent。在 `twotone/ac_stark_ramsey` 量測固定長度 Stark tone 下的 Ramsey 條紋，以 FFT 比較條紋頻率。

**Last updated:** 2026-10-21

## 設定與量測

先完成 readout、qubit frequency 與 pi/2 pulse 校準。確認 `qubit`、`resonator` aliases、接線及各 module。Open／Defaults 捕捉來源建立初值，Run 使用表單上的設定。

- Stark resonator tone 使用 `res_probe` 範本，頻率取 `$resonator.freq`，起始長度為 10 µs。這個長度在 Ramsey delay 掃描中保持固定。
- Ramsey π/2 pulse 優先使用 `$qubit.module.x90`，沒有時使用可編輯的 `pi2_pulse` 範本。請確認其 gain、frequency 與 waveform。
- Stark tone gain (a.u.) 初始範圍為 0 到 0.1，共 21 點。掃描在 gain 平方上等距取點，再做硬體量化。
- Free evolution time (µs) 初始範圍為 0 到 1 µs，共 101 點。Tone settling delay 初值為 0.1 µs。
- Phase-ramp detune 初值為 1 MHz。第二個 pi/2 pulse 的相位增加 360 * detune * Ramsey delay，MHz 與 µs 相乘。
- Reset 可停用。Readout 優先使用 `$resonator.module.readout`，缺少時使用可編輯範本。Reps 和 Rounds 初值為 1000／10，分別是每組 gain／delay 每輪的 shot 數和每個 gain 的平均輪數。Relax delay 初值 1 µs，是每個 shot 後的等待。

每點先 reset，再同時執行 Stark tone 與 Ramsey 序列。Ramsey 序列依序為 settling delay、pi/2、掃描 delay、pi/2。兩者都完成後才 readout。請設定 Stark 長度，使它覆蓋所需的 Ramsey 區段，並檢查 pulse 的 pre/post delay。10 µs 是可編輯的固定初值；請自行設定足以覆蓋序列的安全時長。

Run 左圖的 x 軸是 Stark tone gain，y 軸是兩個 π/2 脈衝間的自由演化時間；色階是以 IQ 的主要變化方向旋轉後取實值，單位 a.u.，不是布居。右圖比較最近兩個 gain 的切線，方便檢查條紋是否可辨。空白代表該區尚未取數。

## FFT 分析

Analyze 的 FFT reference frequency 預設使用本次量測設定的 Phase-ramp detune，缺少量測設定時用 0 MHz。此值可另行修改，只影響分析。分析保留其附近 ±5 MHz 的 FFT 頻率窗，並從顯示頻率軸扣除此值。Stark gain cutoff 設定時排除大於等於該 gain 的資料。

Analyze 上圖沿用 Run 的 gain／時間對比圖。下圖以 gain² 為橫軸，縱軸是相對 FFT reference frequency 的偏移 (MHz)，色階是每個 gain 的 FFT 模長除以自己最大值 (a.u.)，不能跨列比較條紋強度。零偏移表示 FFT 峰等於 reference，不一定是未受 Stark shift。先以最小 gain 當對照，再比較同一條頻率 ridge；例如 reference 1 MHz、ridge 位於 +0.7 MHz 時，該峰是 1.7 MHz。若 ridge 撞到 ±5 MHz 視窗邊界，調整 reference 使它留在圖內。若想分辨間隔 Δf MHz，延遲跨度須達到至少約 1/Δf µs；若最快條紋為 f MHz，將步距設小於 1/(2f) µs 以免 alias，並優先保留每周期約十點。這兩張都是二維圖，不畫逐點 shot SEM。摘要列同次 Reps × Rounds。此入口不擬合 Stark coefficient 或 T2、不寫回，也沒有專用 recipe。

## 保存與載入

Save 保存本次量測設定、gain／delay 軸與二維 complex IQ。記憶體 delay 為 µs，原生檔案為 s。Save／Load 使用原生格式與 tag `twotone/ge/ac_stark_ramsey`。Load 保留目前表單輸入，Analyze 使用載入結果。

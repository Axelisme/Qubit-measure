# Heralded T2 Ramsey

**讀者：** operator、agent。執行單 qubit heralded Ramsey，分析、審閱寫回提案並保存結果。

## 執行前

先準備校準的 π/2 pulse 與單音 pulse readout。Pi/2 Pulse 使用 `$qubit.module.x90`，Readout 使用 `$resonator.module.readout`。缺 slot 時表單提供範本，未校準的必填值保持 Invalid。Mux 不適用。

Readout 必須包含同座標的 circle、linear 或 centroid3 classifier。分類座標是未扣 offset 的 normalized ADC IQ。Circle 需要 g/e centers 與正的 GE Post radius，herald_states 為 g/e/other，other 是拒判區。Linear 為 g/e，沒有拒判區。Centroid3 為 g/e/f，f 不是 other。即時 IQ 與擬合只採用 herald 判為 g 的 shot。Herald 與 signal 使用兩個 PulseReadout instance，共用同一 readout 設定與 classifier。

Reset 是選用的前置 reset，缺 slot 時停用。Herald 本身只做 host post-selection，不做 FPGA feedback 或 active reset。模擬讀出仍受 dispersive validity guard 限制，不能直接套用其他讀出線的 gain 界限。

## 設定與 Run

Open／Reset 捕捉 qubit、resonator 與初值。改來源後用 Reset 重新捕捉。Refresh 只更新來源選項，不覆蓋已編輯輸入。

Delay 初值為 0 到 1.5 × captured qubit.t2r us，共 101 點。Relax delay 初值為 5 × captured qubit.t1 us。缺 T1 時使用 150 us，無效的已存 T1 則讓必要衍生值留空。這些是可編輯的初值，不是硬體安全保證。

Reps 是每點每 round 的 shots，Rounds 是軟體 rounds。Acquire retry 控制暫時性取數錯誤的重試次數。Early-stop SNR 可在一個 round 完成後停止平均，留空則停用。

序列是 optional reset → herald readout → π/2 → delay → phase-shifted π/2 → signal readout。Detune ratio 是每個 delay step 的 fringe 週期數，預設 0.05。實際相位斜率以 ratio / step 換成 MHz，只改最後一個 pulse 的 phase，不改 drive frequency。非零 ratio 搭配零 step 會在取數前報錯。

Run 的 measurement 顯示通過 shot 的平均 IQ 投影。取數層跨 round 依通過數加權。零通過點的 IQ 為 NaN，通過比例為 0。未完成點保留 NaN。

## Analyze 與寫回

Analyze 取 herald-g 的 signal 條件平均 IQ，先排除非有限 delay／IQ，再做 IQ rotation，取實部擬合。本實驗直接依 Fit fringe 的 `fit_fringe` 選擇擬合工具。開啟時用 `fit_decay_fringe` 擬合 decaying fringe，關閉時用 `fit_decay` 擬合純 exponential decay。通過比例不當作 fit weight。分析不需要 saved cfg，有效資料不足或無法擬合時分析會失敗。

Fit fringe 預設開啟。Fit phase offset 的 `fit_phase` 預設開啟，允許自由 fringe phase。關閉時，若最靠近零 delay 的實部訊號高於全曲線最大值與最小值的中點，就固定 phase 為 0°，否則固定為 180°。此選項不影響純 decay。要量純 decay，可將 Detune ratio 設為 0 並關閉 Fit fringe；零 detune 不會自動關閉 Fit fringe。本實驗在 `fit` 圖畫出旋轉後的資料與擬合曲線，標題顯示壽命與標準誤，fringe fit 另顯示頻率與標準誤。

Analyze 回傳 `HeraldingT2RamseyAnalysis`：

- `t2r`、`t2r_err` 是 Ramsey 衰減時間與標準誤，單位 us。
- `detune`、`detune_err` 是觀測到的 fringe 頻率與標準誤，單位 MHz，包含施加的 phase ramp；純 decay 時兩者為 0。
- 以上四個數值欄位皆不為 None。`fit_quality["fit"]` 保存無因次的殘差與相對誤差診斷。型別允許未提供診斷時 `fit_quality` 為 None，本實驗的 Analyze 每次成功都提供診斷。

寫回目標與普通 Ramsey 相同，說明標為 Heralded：
- `$qubit.t2r` 保存 fitted coherence time，單位 us。
- Fringe fit 有 source cfg 與已知 realized detune 時，`$qubit.freq` 提案為 source π/2 frequency + realized detune − fitted fringe detune，單位 MHz。
- 有 captured qubit 與 Run device snapshot 時，另提 captured-bias 的 `t2r_us/t2r_err_us` record。

先審閱提案，再明確 Apply。Frequency 提案只改 qubit.freq，不改現有 pulse library。修改頻率後需重新校準 pulse，選取新 module，再 Reset 下游表單。

## Save、Load 與 MCP

Save 將 `sums` 寫入 native `data.h5`，形狀為 (Nt, classes, 6)，linear 的 classes 為 2，circle 與 centroid3 為 3，後兩軸為 herald_states（依存檔 classifier 順序）與 components（count、ΣI、ΣQ、ΣI²、ΣIQ、ΣQ²）。通過比例只在分析或即時呈現時以 g 計數除以總計數，不另存變數。舊格式缺少 `sums` 時明確報錯。檔案時間使用 seconds，載回恢復 us。Cfg 保存此次 readout classifier。Load 不用目前表單或 classifier 替換歷史資料。

Load 後可再 Analyze。與普通 Ramsey 一樣，native load 不恢復 run-only realized detune，因此不提出 frequency 修正；t2r 與符合來源條件的 record 提案仍可用。

MCP recipe `heralding_t2ramsey` 開啟同一張表單。`points` 對應 Delay Expts，`max_delay_us` 設 stop。`fit_fringe` 與 `fit_phase` 只改 analysis。省略輸入保留表單值，null 原樣交欄位 owner，overrides 最後套用。缺必要值時在 Run 前回報 needs_parameters。Recipe 在成功 Run 後分析、詢問寫回並保存，分析失敗仍保存已取得資料。

## 解讀限制

Pass fraction 是 classifier 接受的比例，不是獨立驗證的 ground purity 或 reset fidelity。Herald 會改變樣本集合，不能只因對比增加就宣稱分類正確。離線 single-fluxonium 驗收不代表硬體 pulse 校準、gate fidelity 或 solver／truncation 收斂。FTF joint-state IQ cluster 不能替代單 qubit g/e classifier。

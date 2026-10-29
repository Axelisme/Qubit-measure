# Heralded T2 Ramsey

**讀者：** operator、agent。執行單 qubit heralded Ramsey，分析、審閱寫回提案並保存結果。

## 執行前

先準備校準的 π/2 pulse 與單音 pulse readout。Pi/2 Pulse 使用 `$qubit.module.x90`，Readout 使用 `$resonator.module.readout`。缺 slot 時表單提供範本，未校準的必填值保持 Invalid。Mux 不適用。

Readout 必須包含 classifier。Linear 使用 angle 與 threshold，angle 的單位是 radians。Circle 使用 g/e centers 與 radius，缺 radius 時先完成 GE Post。分類座標是未扣 offset 的 normalized ADC IQ。Circle 接受 g 圓內且比 e 更近的 shot，Linear 接受 ground 一側。Herald 與 signal 使用兩個 PulseReadout instance，共用同一 readout 設定與 classifier。

Reset 是選用的前置 reset，缺 slot 時停用。Herald 本身只做 host post-selection，不做 FPGA feedback 或 active reset。模擬讀出仍受 dispersive validity guard 限制，不能直接套用其他讀出線的 gain 界限。

## 設定與 Run

Open／Reset 捕捉 qubit、resonator 與初值。改來源後用 Reset 重新捕捉。Refresh 只更新來源選項，不覆蓋已編輯輸入。

Delay 初值為 0 到 1.5 × captured qubit.t2r us，共 101 點。Relax delay 初值為 5 × captured qubit.t1 us。缺 T1 時使用 150 us，無效的已存 T1 則讓必要衍生值留空。這些是可編輯的初值，不是硬體安全保證。

Reps 是每點每 round 的 shots，Rounds 是軟體 rounds。Acquire retry 控制暫時性取數錯誤的重試次數。Early-stop SNR 可在一個 round 完成後停止平均，留空則停用。

序列是 optional reset → herald readout → π/2 → delay → phase-shifted π/2 → signal readout。Detune ratio 是每個 delay step 的 fringe 週期數，預設 0.05。實際相位斜率以 ratio / step 換成 MHz，只改最後一個 pulse 的 phase，不改 drive frequency。非零 ratio 搭配零 step 會在取數前報錯。

Run 的 measurement 顯示通過 shot 的平均 IQ 投影。取數層跨 round 依通過數加權。零通過點的 IQ 為 NaN，`pass_fraction` 為 0。未完成點保留 NaN。

## Analyze 與寫回

Analyze 先排除非有限 delay/IQ，再交普通 Ramsey 的 IQ rotation 與 decay/fringe fitting。`pass_fraction` 不當作 fit weight。有效資料不足或無法擬合時分析會失敗。

Fit fringe 預設開啟。Fit phase offset 預設開啟，允許自由 fringe phase；關閉時由近零 delay 的訊號選固定 0°／180°。要做純 decay，可將 Detune ratio 設為 0 並關閉 Fit fringe。

寫回目標與普通 Ramsey 相同，說明標為 Heralded：
- `$qubit.t2r` 保存 fitted coherence time，單位 us。
- Fringe fit 有 source cfg 與已知 realized detune 時，`$qubit.freq` 提案為 source π/2 frequency + realized detune − fitted fringe detune，單位 MHz。
- 有 captured qubit 與 Run device snapshot 時，另提 captured-bias 的 `t2r_us/t2r_err_us` record。

先審閱提案，再明確 Apply。Frequency 提案只改 qubit.freq，不改現有 pulse library。修改頻率後需重新校準 pulse，選取新 module，再 Reset 下游表單。

## Save、Load 與 MCP

Save 將 `signals` 與同一 time grid 的 `pass_fraction` 寫入 native `data.h5`。檔案時間使用 seconds，載回恢復 us。Cfg 保存此次 readout classifier。Load 不用目前表單或 classifier 替換歷史資料。

Load 後可再 Analyze。與普通 Ramsey 一樣，native load 不恢復 run-only realized detune，因此不提出 frequency 修正；t2r 與符合來源條件的 record 提案仍可用。

MCP recipe `heralding_t2ramsey` 開啟同一張表單。`points` 對應 Delay Expts，`max_delay_us` 設 stop。`fit_fringe` 與 `fit_phase` 只改 analysis。省略輸入保留表單值，null 原樣交欄位 owner，overrides 最後套用。缺必要值時在 Run 前回報 needs_parameters。Recipe 在成功 Run 後分析、詢問寫回並保存，分析失敗仍保存已取得資料。

## 解讀限制

Pass fraction 是 classifier 接受的比例，不是獨立驗證的 ground purity 或 reset fidelity。Herald 會改變樣本集合，不能只因對比增加就宣稱分類正確。離線 single-fluxonium 驗收不代表硬體 pulse 校準、gate fidelity 或 solver／truncation 收斂。FTF joint-state IQ cluster 不能替代單 qubit g/e classifier。

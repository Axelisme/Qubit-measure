# Heralded T2 Echo

**讀者：** operator、agent。執行單 qubit ground-heralded Hahn Echo，審閱 T2e 提案並保存結果。

## 執行前

先準備校準的 Pi Pulse、Pi/2 Pulse 與單音 pulse Readout。Open／Reset 從固定 capture 的 `$qubit.module.x180`、`$qubit.module.x90`、`$resonator.module.readout` 取初值。缺少 slot 時，表單使用範本；範本不是硬體安全預設，必須補齊通道、頻率、gain 與 timing。Delay 預設從 0 掃到 captured T2e 的 1.5 倍，Relax delay 預設為 captured T1 的 5 倍。缺校準時手動填值。

Readout 必須包含同座標的 GE Post circle classifier，具有 g/e centers 與 radius。分類座標是未扣 offset 的 normalized ADC IQ。Linear、centroid3、Mux 與其他 qubit 的 cluster 不適用。

兩次讀出使用同一組 pulse／ADC 與 classifier。這是 host post-selection，不是 FPGA feedback，沒有 active reset 或 joint herald。模擬仍受 dispersive validity guard 限制，不能把另一條線的 readout gain 當作安全界限。

## Run

可選 Reset 在 herald 前執行。序列依序為 Reset → herald readout → π/2 → half delay → π → half delay → final π/2 → signal readout。Reset 留空時不發 reset pulse。取數保存 herald 判為 g、e、other 各類的 count 與 IQ 總和，即時圖與擬合只取 g 類的 signal 條件平均。

Delay 是總自由演化時間，單位 us，不是半延遲。兩個 half delay 各自做硬體時間量化。Detune ratio 是每個總延遲 step 的 fringes；0 關閉相位 ramp。非零時，表單轉成 ratio／requested step 的 MHz，再隨總 delay 增加 final π/2 的 phase，不改 drive frequency。退化成零 step 的 sweep 不能使用非零 ratio。

Reps 是每點每 round 的 shots，Rounds 是軟體平均輪數，Relax delay 是每 shot 結束後的等待。Acquire retry 控制暫時取數失敗的重試，Early-stop SNR 留空時不提前停止。

Measurement 顯示篩選後平均 IQ 的投影。跨 round 依通過數加權，不直接平均各 round 的平均值。零通過點的 IQ 為 NaN，通過比例為 0；未完成點保留 NaN。

## Analyze 與 writeback

Analyze 先排除非有限 time／IQ，再交普通 Echo owner。Owner 略去剩餘資料的第一點，再做 IQ rotation 與 decay 或 fringe fit。Fit method 的 decay 使用 exponential envelope；fringe 使用 decaying fringe，Fit phase 只控制 fringe 的 phase offset。零 detune 強制 decay。有效點不足時分析失敗，不改用另一套擬合。Pass fraction 是診斷，不是擬合權重。

寫回與普通 Echo 相同：提出 `$qubit.t2e`，有固定 capture 與 device snapshot 時，也提出同一 qubit 的 bias record。描述標明 Heralded。T2e 必須為正且有限，error／T2e ≤ 0.5、R² ≥ 0.9、normalized RMS ≤ 0.1，才預設勾選；缺 metrics 或其他不符會顯示 review note。先看 fit 與提案，再明確 Apply；Analyze 不會自行寫回。

## Save 與 Load

Save 在 native `data.h5` 保存 `sums`，形狀為 (Nt, 3, 3)，後兩軸為 herald_states（g、e、other）與 components（count、ΣI、ΣQ）。通過比例只在分析或即時呈現時從計數計算，不儲存。舊格式缺少 `sums` 時明確報錯。檔案時間使用 seconds，Result 載回為 us。Cfg 保留本次 pulses、detune 與 readout classifier。從本實驗 Load 後可再 Analyze，不用目前表單的 classifier 或 aliases 補歷史設定。

通過比例是 classifier 接受的 shot 比例，不是獨立的 ground purity、reset fidelity 或 gate fidelity。Herald readout 本身可能改變狀態。離線的有限溫度與 Markov bath 模型不能認證低頻噪聲 refocusing，流程走通也不代表硬體 T2e 已校準。

**讀者：** operator、agent。用共同頻率偏移掃描 mux 共振腔，找出共振頻率與線寬。

# Mux onetone

同時量多個 resonator 的複數平均 IQ；擬合給出共振 RF、半高全寬 κ 和背景相位等參數。

**序列：** 每個偏移點先做共用 reset，再送 mux readout 並讀取選定的 ADC。

## 關鍵參數

將 `mux` alias 綁到 `group/mux`。完整 **Readout** 的 tone、resonator 與 ADC 依位置對應。只量 resonator 時可不綁 qubits；若提供，qubit 必須與 resonator 依序配對。**Participants** 用不重複的原始零起算序號指定量測順序；`[2,0]` 先 R3 後 R1。缺少、未綁或空的 `participants` alias 預選全組；編輯選取不重設 tones。

**Frequency offset** 是所有選定 tone 共用的 MHz 偏移，不是絕對頻率。核對每個 tone 的基準 RF，選範圍覆蓋共振兩側的背景。即使 tone 不參與，仍須確認完整 **Readout** 的頻率與 ADC 身分；Run 前核對 mixer、接線、功率和 ADC 窗。

### 無 readout slot 的候選來源

- **Readout** 沒有 slot 時，generator 與 ADC 取組別接線，tone RF 取 resonator 校準，gain 初填 0.1；**Mixer freq** 取 group 的 mixer 校準（缺值則留空必填），**RO length** 固定初填 1 µs。
- 共用等待 `w = 5/(2π·min κ)` µs，κ 用 MHz。**Trig offset** 取 group flight time + w，**Length** 取 trigger + 積分窗。缺 flight time 或任一 κ 就留空 trigger 和 length；只有 κ 都有值時才把 **Post delay** 初填 w，否則保留 0 µs。
- 有 slot 則沿用其 mixer 和 ADC 窗，只用已知 κ 算 w 加到原 trigger，必要時延長 pulse；**Post delay** 取原值與 w 較大者。沒有可用 κ 則不加等待。這些都是初值，不是路徑校準。

**Model type** 選 hanger、transmission 或 auto。**Electrical delay seed (µs; blank = auto)** 留空時，各 tone 使用與 generator／ADC 路徑相符的校準 seed，無匹配才全域搜尋（上限 100 µs）；明填有限值則共用該 seed，仍擬合 delay。

在 `mux_onetone` 中，省略保留表單與分析設定。Null `participants`、`df`（清除掃描端點）、`n`、`reps`、`rounds` 若未用 override 補齊，會在 Run 前的 readiness 檢查報錯。Null `model_type`、`fit_bg_amp_slope` 或 `fit_bg_phase_curvature` 會在採集後的 Analyze 被拒絕；null `edelay` 選自動 seed。見 [mux recipe 輸入](../../../recipes/mux-guide.md#mux_onetone)。

## 怎麼讀圖

Live 橫軸為基準 RF + 名目 offset (MHz)，縱軸為平均 IQ 幅度，不是布居。Analyze 去掉首尾點，幅度圖的標線指共振中心；用相位與 IQ 圖確認同一模型也符合複數軌跡。頻譜不畫逐點 shot SEM；± 是 fit 參數誤差，R² 是解釋變異的比例，正規化殘差 RMS 是相對訊號範圍的偏差。以原始 tone 序號辨認 `fit_i`；label 可能重複。

## 下一步

Writeback 更新成功 tone 的 readout RF，另提 readout slot 與對應 resonator 的 freq、κ、背景相位和適用的路徑 delay；失敗或未量 tone 原值不變。完整 resonator 數和 readout 數不符時沒有逐 resonator 提案。重複目的地須只選一列。Load 缺歷史 cfg 時無法 FIT；沒有該 Run/Load 的參數來源時可檢查結果，但不能套用候選。

用這份 readout 做 herald 前，從 `post_delay = 5/(2π·min κ)` µs 起步（κ 用 MHz，等待從 readout 波形結束起算）。等待太短時，殘留腔光子會使名目 Xπ 失準。Herald 後打名目 Xπ，保持其餘設定不變，和更長等待比較終端 e 比例；若比例仍上升，先調整等待再做 heralded 掃描。FIT 不會校準這段等待。共用操作見 [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md)。

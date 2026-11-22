---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0083-program-macros-and-timelines.md
  - ../../lib/zcu_tools/experiment/README.md
  - ../../lib/zcu_tools/simulate/adc/README.md
  - ../../lib/zcu_tools/simulators/ftf/environment/README.md
  - 2026-11-21-ftf-legal-readout-timing.md
basis:
  - "本 repo merge base 163fbeb51d62ba21c671915d8bad584781a192bf"
  - "配對候選 19897d3464c7f7a5d9b3d7e69145525f4f37c756；誤差表候選 a8ece4a3d7089a9dca7abce793ffc0e9ea86dbea；review 修後候選 3480a4ffe713178b7ad731affcca70c38c815d66"
  - "外部來源：無"
---

# FTF 讀出窗依裝置時間配對

## 結論

FTF 的讀出配對改用每個 receiver 的裝置時間視窗。ADC 透過 `adc_to_device_time_ns(adc_time_ns, *, time_of_flight_ns)` 擁有唯一的 `t−τ` 換算；ADC solver 與 planner 使用同一函式。ADC 的取樣格點、相位、trigger 順序與解調都留在 ADC 時間。

每個視窗分別與包含 ringdown 的 block interval `[start,next_start)` 判交集，最後相交的 block 是 owner；最後一個 block 延伸至 point 結束。每個視窗保留自己的交集集合。同一 point、同一 ADC 有多個相同 owner 的視窗時，有場事件使用這些視窗相交 blocks 的聯集，每個 block 只收一次。其他 ADC 或 owner 不會提供 pulse member。沒有入射 pulse 的事件只回傳 receiver offset 與 noise。完全早於第一個 block 的視窗使用該 ADC 沒有 field grids 的既有事件，不新增 projection。既有事件 layout 無法表示時，planner 精確失敗，不移動 grid 或 projection。

投影塌縮仍在讀出 pulse 的裝置時間起點。跨投影的視窗仍以一個 owner 狀態套用全窗。FTF 的每個 ADC 事件仍從 vacuum 起始。這兩項都是近似。Vacuum-start 誤差與 owner-state 誤差分開量測；I/Q 數字及其限制見 [FTF environment README](../../lib/zcu_tools/simulators/ftf/environment/README.md#known-approximations)。

planner 修正時一併移除了漏配在途 pulse 的過渡守門及只服務該守門的拒絕測試，並以公開在途行為的正向測試取代。移除邏輯時同步移除只服務它的測試，這項原則不變。

## 問題

Planner 與 ADC solver 對同一 receiver delay 使用了兩種描述。ADC solver 在 `t−τ` 評估物理輸出，並保留 ADC sample time；planner 卻按未平移的 ADC 時間配對 trigger，只用 pulse 抵達時間處理較晚發射的 pulse。合法 lookback 可能在建立內容前被拒。較早且仍在途的 pulse 可能漏出事件，因此 FTF guard 拒絕計畫，而不是回傳缺少 I/Q 訊號的結果。

一個 `chain_length` 0.2 µs readout window 的離線 point 觸發了在途守門。這表示合法 caller 無法完成規劃，不能證明修正配對會縮短 Run。1.2 與 2.2 µs 的 Run 在改動前也分別超過時間限制，因此沒有從配對修正推論效能提升。

第二個問題與配對無關：ADC 事件從 vacuum 起始，沒有帶入前一個事件的 cavity 殘場。對保存的 R1 fixture 以零 post-delay 做單模估算，下一個視窗起點漏掉的場振幅約為穩態振幅的 9.5%，窗末約為 0.4%。這是場振幅估算，不是 I/Q 誤差量測。

使用者要求將 ADC observation window 往前移，而不是移動 pulse emission time。FTF receiver 的 `time_of_flight_ns` 與 form 的 `time_of_fly` 是不同輸入。模擬器使用 receiver 設定的 delay；form 值仍用於 program trigger offset。

## 考慮過的選項與比較

**事件配對與 R5。**

- **保留 emission-time 配對（K）。** 這避免了程式變更，但合法 lookback 仍會被拒絕，在途缺 pulse 仍只能被 guard 擋下，且跨事件殘場仍缺失。
- **以 device-time window 配對（F，採用）。** 對 receiver `r`，將 ADC 視窗 `[t_first,t_last]` 換成 `[t_first−τ_r,t_last−τ_r]`，再與裝置時間的 block interval 比交集。ADC 數值路徑仍使用原取樣格點和相位，τ 不會套兩次。這修正 lookback 與在途 pulse 的成員歸屬，但不傳遞前事件的腔場。
- **同一批加入 history-specific continuous cavity field（N）。** 這會改變共用的 `ReadoutEvent`／`ReadoutBatch` 及 FTF history-to-ADC handoff，牽動多個 ADC 路徑與 simulator caller。當時沒有先量出 R5 的 I/Q 效應，也沒有低成本的 field handoff 方案，因此先保留 vacuum-start 近似，量完再決定。

**視窗跨投影時的狀態。** 當時 `rpi_length` 使用固定 `0.11 µs` 長度餘量，裝置時間視窗可跨過投影 block 起點。比較三種做法：整窗使用一個 owner state；按投影點切分視窗並組 joint probability；或跨界時拒絕。採用整窗使用最後相交 block 的 owner state，符合使用者直接選擇的事件層近似。分段需要另定聯合態與投影路徑；拒絕則會擋下當時的 `rpi_length` 與零 post-delay 的 chain。投影時間留在 pulse start，不移到 `trigger−τ`。

**換算 API。** 換算函式只收 scalar delay，不收完整 `Receiver`：planner 在組出經校準的 ADC receiver 前，只有 channel-to-delay mapping。`time_of_flight_ns` 是 keyword-only；scalar ADC time 回傳 `float`，float64 array 回傳同形新 array；負的裝置時間合法，delay 必須有限且非負。函式不重掃 sample array；時間有效性由 `SampleGrid` 與 `TimeBase` 負責。

**Block interval 與早窗。** 若只看 pulse support，純 ringdown 視窗會變成空事件。`[start,next_start)` interval 保留既有 block ownership 到 ringdown 結束。完全早到的視窗產生 receiver-only baseline，可使用該 ADC 沒有 field grids 的既有事件，且不建立 projection。若沒有事件 layout 可表示，planner 回報 ADC、trigger 與原因。

**量測。** 近似比較使用固定 conditional histories 與相同 ADC grids。Scratch hook 先在 vacuum/no-switch 控制組中比對未掛 hook 的公開 response，再核對場的連續性，之後才注入初始場或分段狀態。量測值只適用於這些模型條件，不代表硬體誤差。結果表已寫入 FTF environment README。

## 被否決的選項與理由

- 將 pulse support 往後移 `τ`，在單一 scalar receiver delay 下幾何上可能等價，但會讓 planner 形成第二個換算 owner，並有重複套用 delay 的風險。因此換算仍由 `simulate.adc` 擁有。
- 保留 emission-time matcher 與 guard，會繼續讓合法 lookback 沒有結果，也會把過渡性的漏 pulse 拒絕留作最終契約。
- 同一變更傳遞完整 history-specific field，會在量出漏場的 I/Q 差異或選定低成本表示法之前擴大共用公開 event 邊界。
- 每個 projection 都切分 window，需要既有 event model 尚未定義的 joint probability 與 field/state composition。
- 在 projection boundary 拒絕 window，會擋下當時的 `rpi_length` 配置，即使該配置合法。
- 隨 `τ` 移動 projection time，會混淆 ADC 記錄延遲與裝置端 backaction。使用者表示目前沒有 active reset 設計；未來若加入依讀出結果回饋的 active reset，必須重新檢視往返延遲。
- 只刪 guard test 卻保留程式，或讓舊 guard test 一直紅到後續 ticket，都會讓行為與測試分離，或留下預期紅燈。改配對的 ticket 同時移除 guard 與只服務 guard 的測試，再以公開正向行為測試取代舊拒絕預期。

## 留下的問題

事件仍從 vacuum 起始，因此不會跨事件傳遞 cavity field。固定歷史的 `chain/t1` 比較中，零 post-delay 的最大 `‖ΔIQ‖₂` 為 845.841 ADC sample-sum units，`5/κ` 時為 69.505；在該 fixture 中分別是 receiver-noise RMS 的 0.214 與 0.0175。這些數字尚未決定是否在其他 task 加入低成本的 field carry。

跨投影的 window 仍以一個 owner state 套用全窗。固定 `0.11 µs` 長度餘量的 `rpi_length` 對照中，最大差為 1,127.855 sample-sum units，是參考 g/e separation 的 9.62 倍、receiver-noise RMS 的 0.636 倍。這不是完整 Run 的分類估計。

兩種比較都沒有量測硬體誤差，也沒有隔離[同 owner 的 pulse 聯集](2026-11-21-shared-owner-readout-membership.md)的貢獻。後來的表單改從讀出模組取得長度餘量，見 [`rpi_length` 長度差](2026-11-21-rpi-length-fixed-margin.md)；上述數字不代表新的表單預設。Active reset、receiver group-delay dispersion 與硬體 phase epoch 都在此模型範圍外；若加入這些需求，需另做物理與介面審查。

## 依據

使用者於 2026-11-21 授權的原話是：「读书时间，这种东西，你只要能确保计算是正确的，然后机制是统一的，好维护，你知道存在这个东西，而且不会造成什么 bug 之类的就好了。那具体上做细节，我没什么意见。」 使用者也說：「R5 的话，这个算是模拟的一种误差。老实说，我不确定这是不是可以低成本解决的问题。比如说，可能只记录剩余光指数之类的，我不是很确定有没有好解法。 然后跨2个投影点的话，我觉得 A 就好了。没错，投影时刻只影响 ADC。 Qubit 不影响，这个是合理的，因为就算去程到 Qubit 有 delay， 那对于非 active reset 的操作来说是没有差的。我目前也没有设计 active reset。」

換算 owner 遵循 [ADR-0078](../adr/0078-evolution-module-boundary.md)；ADC 獨立 timeline 遵循 [ADR-0083](../adr/0083-program-macros-and-timelines.md)；receiver delay 與 ADC sample 語意記載於[ADC README](../../lib/zcu_tools/simulate/adc/README.md)。Block 配對與模組 owner 記載於[experiment README](../../lib/zcu_tools/experiment/README.md)；已知近似與數值條件記載於[FTF environment README](../../lib/zcu_tools/simulators/ftf/environment/README.md)。

原始討論和離線收據未入庫。幾何、估算、比較數字與限制見本文及連結的模組文件。
